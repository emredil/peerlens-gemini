"""SymPy Verifier (Gemini port): symbolic verification of claims via generated snippets."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient
from peerlens.agents.equation_extractor import extract_equations_for_claim


@dataclass
class SymPyFinding:
    claim_id: int
    location: str
    claim_type: str
    verdict: str
    severity: str
    detail: str
    generated_code: str
    stdout: str
    stderr: str
    runtime_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    equations_available: list[str] = field(default_factory=list)


@dataclass
class SymPyReport:
    doi_label: str
    n_findings: int = 0
    n_verified: int = 0
    n_refuted: int = 0
    n_unable: int = 0
    findings: list[SymPyFinding] = field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0


SNIPPET_TEMPLATE = textwrap.dedent('''\
    import json, sys, traceback

    import sympy
    from sympy import (
        symbols, Symbol, Function, sqrt, exp, log, sin, cos, tan,
        simplify, expand, factor, together, cancel, trigsimp, radsimp,
        Sum, Product, Rational, oo, pi, I, E, Abs, Piecewise, Eq,
        limit, series, diff, integrate, solve, Matrix, eye, zeros,
        sympify,
    )
    from sympy.parsing.latex import parse_latex
    from sympy.physics.quantum import Commutator, Operator, HermitianOperator
    from sympy.physics.paulialgebra import Pauli

    def _emit(verdict, detail):
        print(json.dumps({"verdict": verdict, "detail": str(detail)[:2000]}))

    try:
    __BODY__
    except Exception as e:
        _emit("unable_to_verify", f"snippet raised: {type(e).__name__}: {e}")
''')


_ALGEBRA_MARKERS = (
    "simplify(", "expand(", ".equals(", "sympify(", "parse_latex(",
    "Commutator(", ".doit(", "limit(", "series(", "solve(",
    ".subs(", "factor(", "cancel(", "together(", "trigsimp(",
    "Eq(", "== 0", "!= 0",
)


def _snippet_did_real_work(snippet: str) -> bool:
    return any(m in snippet for m in _ALGEBRA_MARKERS)


def _wrap_snippet(body: str) -> str:
    indented = "\n".join(("    " + line) if line.strip() else line for line in body.splitlines())
    return SNIPPET_TEMPLATE.replace("__BODY__", indented)


def run_snippet(code: str, timeout: float = 10.0) -> tuple[str, str, str, float]:
    wrapped = _wrap_snippet(code)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        dt = time.time() - t0
    except subprocess.TimeoutExpired:
        return "unable_to_verify", f"timed out after {timeout}s", "", time.time() - t0

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    verdict, detail = "unable_to_verify", "no verdict emitted"
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
                if d.get("verdict") in ("verified", "refuted", "unable_to_verify"):
                    verdict = d["verdict"]
                    detail = d.get("detail", "")
                    break
            except json.JSONDecodeError:
                continue

    combined = f"[stdout]\n{stdout}\n[stderr]\n{stderr}"
    return verdict, detail, combined, dt


SNIPPET_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "snippet": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["reasoning", "snippet", "confidence"],
}


SNIPPET_SYSTEM = """You are a symbolic mathematics assistant. You translate physics
claims about equations into short Python snippets that use SymPy to test them.

You have access to a pre-imported environment (already in scope):
  - sympy names: symbols, Symbol, simplify, expand, Eq, limit, series, diff,
    integrate, solve, Matrix, sqrt, exp, log, oo, Rational, sympify, ...
  - sympy.parsing.latex.parse_latex  (best-effort, may fail on operator math)
  - sympy.physics.quantum: Commutator, Operator, HermitianOperator
  - sympy.physics.paulialgebra: Pauli
  - _emit(verdict, detail) function — call EXACTLY once

Do NOT include import statements. Do NOT do file or network I/O.
Keep the snippet under 60 lines.

You will receive the ACTUAL LaTeX of the equations referenced by the claim
(extracted from the PDF). Use them. Do not guess the algebraic form.

STRATEGY BY CLAIM TYPE:

- reduction ('X reduces to Y when β=0'):
    Read the LaTeX of X and Y from the input. Build both as scalar sympy
    expressions if possible (treat operator symbols as commutative Symbols
    when the reduction only kills additive/multiplicative terms).
    Substitute the parameter and simplify(X_sub - Y). If simplify returns 0,
    verdict='verified'.

- limit / algebraic_identity, scalar case:
    Substitute symbolic values, simplify LHS - RHS. If 0 → verified.
    If simplify returns a non-zero expression that CANNOT be zero for all
    parameter values, verdict='refuted' with the counterexample in detail.

- algebraic_identity, operator case ([A, B] = 0):
    If SymPy's non-commutative support isn't strong enough for the paper's
    algebra (typical for spin chain generators with custom relations),
    emit 'unable_to_verify' explaining what's missing.

- symmetry, conservation:
    Same pattern as operator case above.

- dimensional:
    Not your job — emit 'unable_to_verify'.

MANDATORY FOR 'verified' VERDICT:
Your snippet MUST call at least one of these decisively before emitting
'verified': simplify, expand, .equals, sympify comparison to 0,
Commutator.doit, limit, solve, or a boolean check like `result == 0`.
Prose in comments then 'verified' is not allowed — the sandbox will demote
that to 'unable_to_verify'.

AUTHORITATIVE SOURCE RULE:
The LaTeX equations shown are extracted directly from the PDF and are the
ground truth. The natural-language "claim" is an upstream extractor's
paraphrase and may be imprecise.
  - Build your SymPy expressions from the LaTeX, not from the paraphrase.
  - If the LaTeX and the paraphrase describe different formulas, do NOT
    rewrite the LaTeX to match the paraphrase. Test what the LaTeX says.
    If the paraphrase contradicts the LaTeX, emit 'refuted' AND include
    'Paraphrase-LaTeX mismatch: ...' in detail.

If translation is genuinely impossible from the LaTeX given, emit
'unable_to_verify' honestly. Prefer honesty over false positives."""


def build_snippet(
    claim: dict,
    symbols_catalogue: list[dict],
    equation_latex: dict[str, str],
    client: GeminiClient,
) -> tuple[str, str, float, int, int]:
    model = CONFIG["gemini"]["model_classifier"]

    sym_view = "\n".join(
        f"  {s['symbol']}: {s['meaning']} — units: {s['units']}"
        for s in symbols_catalogue[:20]
    )

    if equation_latex:
        eq_view = "\n".join(f"  Eq. ({num}):  {latex}" for num, latex in equation_latex.items())
    else:
        eq_view = "  (no equation LaTeX was extractable from the PDF)"

    user = f"""Claim to test:
  type: {claim['claim_type']}
  location: {claim['location']}
  referenced equations: {claim.get('referenced_equations', [])}
  natural language: {claim['natural_language']}
  check hint: {claim['check_hint']}

EQUATIONS (actual LaTeX extracted from the PDF):
{eq_view}

Available symbols in this paper (with declared units):
{sym_view}

Use the actual LaTeX above to build the test. If the LaTeX gives you a
concrete algebraic form, translate it to SymPy and test it. If translation
is impossible (essential non-commutative operator structure), emit
'unable_to_verify' honestly.

Propose a SymPy test via the tool."""

    result = client.structured_call(
        model=model,
        system=SNIPPET_SYSTEM,
        user=user,
        tool_name="propose_sympy_test",
        tool_description="Propose a Python snippet using SymPy that decides whether a claim holds.",
        tool_input_schema=SNIPPET_SCHEMA,
        max_output_tokens=2000,
        fallback_models=CONFIG["gemini"].get("fallback_chain", []),
    )

    inp = result.tool_input
    return (
        inp.get("snippet", "_emit('unable_to_verify', 'empty')"),
        inp.get("reasoning", ""),
        float(inp.get("confidence", 0.0)),
        result.input_tokens,
        result.output_tokens,
    )


def verify_paper(
    claims_data: dict,
    pdf_path: Path,
    max_claims: Optional[int] = None,
    client: GeminiClient | None = None,
) -> SymPyReport:
    if client is None:
        client = GeminiClient()

    doi = claims_data.get("doi_label", "")
    report = SymPyReport(doi_label=doi)

    sympy_claims = [
        (i, c) for i, c in enumerate(claims_data.get("claims", []))
        if c.get("verification_method") == "sympy"
    ]
    if max_claims:
        sympy_claims = sympy_claims[:max_claims]

    symbols_catalogue = claims_data.get("symbols", [])

    for claim_id, claim in sympy_claims:
        ref_eqs = claim.get("referenced_equations", [])
        equation_latex: dict[str, str] = {}
        if ref_eqs:
            try:
                lookup = extract_equations_for_claim(doi, pdf_path, ref_eqs)
                equation_latex = {num: eq.latex for num, eq in lookup.items()}
            except Exception:
                equation_latex = {}

        snippet, reasoning, conf, in_t, out_t = build_snippet(
            claim, symbols_catalogue, equation_latex, client,
        )
        report.total_tokens_in += in_t
        report.total_tokens_out += out_t

        verdict, detail, combined_io, runtime = run_snippet(snippet)

        if verdict == "verified" and not _snippet_did_real_work(snippet):
            verdict = "unable_to_verify"
            detail = (
                "Verifier demoted to unable_to_verify: the generated snippet "
                "did not invoke any symbolic-algebra primitive. Original detail: " + detail
            )

        if verdict == "refuted":
            severity = "critical"
        elif verdict == "verified":
            severity = "info"
        else:
            severity = "warning" if conf > 0.5 else "info"

        finding = SymPyFinding(
            claim_id=claim_id,
            location=claim.get("location", ""),
            claim_type=claim.get("claim_type", ""),
            verdict=verdict,
            severity=severity,
            detail=f"{reasoning}\n{detail}",
            generated_code=snippet,
            stdout=combined_io[:2000],
            stderr="",
            runtime_seconds=runtime,
            input_tokens=in_t,
            output_tokens=out_t,
            equations_available=sorted(equation_latex.keys()),
        )
        report.findings.append(finding)

    report.n_findings = len(report.findings)
    report.n_verified = sum(1 for f in report.findings if f.verdict == "verified")
    report.n_refuted = sum(1 for f in report.findings if f.verdict == "refuted")
    report.n_unable = sum(1 for f in report.findings if f.verdict == "unable_to_verify")
    return report


def save_report(report: SymPyReport, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "sympy_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
