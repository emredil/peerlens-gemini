"""Dimensional Checker (Gemini port): two-tier unit-consistency verifier."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import fitz

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


@dataclass
class DimensionalFinding:
    claim_id: int
    location: str
    verdict: str
    severity: str
    evidence: str
    suggested_fix: str
    confidence: float
    verifier_tier: str
    referenced_symbols: list[str] = field(default_factory=list)


@dataclass
class DimensionalReport:
    doi_label: str
    n_findings: int = 0
    n_critical: int = 0
    n_warning: int = 0
    n_info: int = 0
    findings: list[DimensionalFinding] = field(default_factory=list)
    tier_a_tokens_in: int = 0
    tier_a_tokens_out: int = 0
    tier_b_tokens_in: int = 0
    tier_b_tokens_out: int = 0
    tier_b_pages_analyzed: int = 0


TIER_A_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "integer"},
                    "location": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["consistent", "inconsistent", "unable_to_verify", "convention_dependent"],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "warning", "info"],
                    },
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "confidence": {"type": "number"},
                    "needs_vision": {"type": "boolean"},
                    "referenced_symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["claim_id", "location", "verdict", "severity",
                             "evidence", "confidence", "needs_vision"],
            },
        },
    },
    "required": ["findings"],
}


TIER_A_SYSTEM = """You are a rigorous physics referee focused on DIMENSIONAL / UNIT consistency.

You are given a paper's symbol catalogue (each symbol with its stated units)
and a list of claims the extractor has flagged. Your job is to check whether
the units close correctly.

BE STRICT BUT REALISTIC. Physics papers commonly use conventions where
ħ = k_B = c = 1 or absorb dimensional factors into coupling constants. If a
symbol is declared 'dimensionless' but appears alongside energies, this may
be a natural-units convention, not an error — in that case the verdict is
`convention_dependent` (severity info), NOT `inconsistent`.

CHECK PATTERNS:
1. Sum of terms: every term in a sum must share the same units.
2. Equation LHS = RHS units.
3. Argument of exp, log, sin, cos MUST be dimensionless.
4. Products with declared coupling constants: check dimensions close.
5. Symbol reuse: if a symbol is defined with different units in two places,
   flag as inconsistent.

VERDICT GUIDANCE:
- `consistent`: units close and no ambiguity.
- `inconsistent`: units clearly don't close and no natural-units interpretation.
- `unable_to_verify`: you need to see the actual equation to decide → set
  needs_vision=true.
- `convention_dependent`: units appear off, but a standard natural-units
  convention would explain it. Note the convention needed.

SEVERITY:
- `critical`: a real dimensional error that would fail refereeing.
- `warning`: possibly wrong or possibly convention; author should clarify.
- `info`: purely descriptive.

Produce 3-10 findings per paper. If the paper has no dimensional issues at
all, produce a single finding with verdict `consistent` summarising it.

Use LaTeX notation for any mathematical symbols and equations in your output. Write `$\\varepsilon$` not `ε`, `$\\mu$` not `µ`, `$\\pi$` not `π`. Every mathematical symbol must be wrapped in `$...$`. Never use Unicode math characters."""


TIER_B_SYSTEM = """You are a physics referee reading an equation directly from a PDF page image.

You have a specific dimensional question raised by an earlier text-only pass.
Your job: look at the equation the question is about, and give a decisive
verdict.

Return one dimensional finding via the record_dimensional_findings tool with:
- verdict: consistent / inconsistent / convention_dependent
- severity: critical / warning / info
- evidence: cite the specific equation text you see and the specific units.
- suggested_fix: if not consistent.
- confidence: 0.8-1.0 for vision-based verdicts.

Use LaTeX notation for any mathematical symbols and equations in your output. Write `$\\varepsilon$` not `ε`, `$\\mu$` not `µ`, `$\\pi$` not `π`. Every mathematical symbol must be wrapped in `$...$`. Never use Unicode math characters."""


def _format_extractor_output(claims_data: dict) -> str:
    symbols_list = "\n".join(
        f"  [{i}] {s['symbol']}: {s['meaning']} — units: {s['units']} "
        f"(first seen: {s.get('first_seen_at', '')})"
        for i, s in enumerate(claims_data.get("symbols", []))
    )
    claims_list = []
    for i, c in enumerate(claims_data.get("claims", [])):
        claims_list.append(
            f"  [{i}] type={c['claim_type']}, method={c.get('verification_method', '?')}, "
            f"loc={c['location']}, refs={c.get('referenced_equations', [])}\n"
            f"      claim: {c['natural_language']}\n"
            f"      check_hint: {c['check_hint']}"
        )
    return f"""PAPER: {claims_data.get('doi_label', '')}

SYMBOL CATALOGUE:
{symbols_list}

CLAIMS TO CONSIDER:
{chr(10).join(claims_list)}"""


def tier_a_check(claims_data: dict, client: GeminiClient | None = None, uploaded_file: Any = None) -> tuple[list[DimensionalFinding], int, int, list[dict]]:
    if client is None:
        client = GeminiClient()
    model = CONFIG["gemini"]["model_classifier"]

    result = client.structured_call(
        model=model,
        system=TIER_A_SYSTEM,
        user=_format_extractor_output(claims_data),
        tool_name="record_dimensional_findings",
        tool_description="Record dimensional (unit) consistency findings for a paper.",
        tool_input_schema=TIER_A_SCHEMA,
        max_output_tokens=4096,
        fallback_models=CONFIG["gemini"].get("fallback_chain", []),
        uploaded_file=uploaded_file,
    )

    raw = result.tool_input.get("findings", [])
    findings = [DimensionalFinding(
        claim_id=f["claim_id"],
        location=f["location"],
        verdict=f["verdict"],
        severity=f["severity"],
        evidence=f["evidence"],
        suggested_fix=f.get("suggested_fix", ""),
        confidence=f["confidence"],
        verifier_tier="text_only",
        referenced_symbols=f.get("referenced_symbols", []),
    ) for f in raw]

    need_vision = [f for f in raw if f.get("needs_vision")]
    return findings, result.input_tokens, result.output_tokens, need_vision


def _pdf_page_to_bytes(pdf_path: Path, page_num: int, zoom: float = 2.0) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()


def _find_equation_page(pdf_path: Path, eq_number: str) -> Optional[int]:
    doc = fitz.open(pdf_path)
    try:
        needle = f"({eq_number})"
        for i, page in enumerate(doc):
            if needle in page.get_text("text"):
                return i
    finally:
        doc.close()
    return None


def tier_b_verify(
    doi_label: str,
    pdf_path: Path,
    raw_finding: dict,
    claims_data: dict,
    client: GeminiClient | None = None,
    uploaded_file: Any = None,
) -> tuple[Optional[DimensionalFinding], int, int, int]:
    if client is None:
        client = GeminiClient()

    claim_id = raw_finding.get("claim_id", -1)
    eq_refs: list[str] = []
    if 0 <= claim_id < len(claims_data.get("claims", [])):
        eq_refs = claims_data["claims"][claim_id].get("referenced_equations", [])

    page_num = None
    for eq in eq_refs:
        stripped = eq.strip("()")
        page_num = _find_equation_page(pdf_path, stripped)
        if page_num is not None:
            break

    if page_num is None:
        return None, 0, 0, 0

    png_bytes = _pdf_page_to_bytes(pdf_path, page_num)

    result = client.structured_call(
        model=CONFIG["gemini"]["model_specialist"],
        system=TIER_B_SYSTEM,
        user=(
            f"The text-only pass raised this dimensional question:\n\n"
            f"Location: {raw_finding.get('location', '')}\n"
            f"Evidence so far: {raw_finding.get('evidence', '')}\n"
            f"Referenced equations: {eq_refs}\n\n"
            f"Look at the equation(s) on this page and decide. Report via the tool."
        ),
        tool_name="record_dimensional_findings",
        tool_description="Record a dimensional finding based on the equation page shown.",
        tool_input_schema=TIER_A_SCHEMA,
        max_output_tokens=2048,
        images=[{"mime": "image/png", "data": png_bytes}],
        fallback_models=CONFIG["gemini"].get("fallback_chain", []),
        uploaded_file=uploaded_file,
    )

    raw_out = result.tool_input.get("findings", [])
    if not raw_out:
        return None, result.input_tokens, result.output_tokens, 1

    f = raw_out[0]
    finding = DimensionalFinding(
        claim_id=raw_finding.get("claim_id", -1),
        location=f.get("location", raw_finding.get("location", "")),
        verdict=f["verdict"],
        severity=f["severity"],
        evidence=f["evidence"],
        suggested_fix=f.get("suggested_fix", ""),
        confidence=f["confidence"],
        verifier_tier="vision",
        referenced_symbols=f.get("referenced_symbols", []),
    )
    return finding, result.input_tokens, result.output_tokens, 1


def check_paper(
    claims_data: dict,
    pdf_path: Path,
    max_vision_calls: int = 3,
    client: GeminiClient | None = None,
    field: str = "physics",
    uploaded_file: Any = None,
) -> DimensionalReport:
    if client is None:
        client = GeminiClient()

    doi = claims_data.get("doi_label", "")
    report = DimensionalReport(doi_label=doi)

    # Field gate: dimensional analysis is meaningful for physical quantities
    # with units. Math papers deal with dimensionless topological / algebraic
    # objects. Biology papers describe processes with parameters whose
    # dimensional consistency is rarely the referee's primary concern.
    # Chemistry and physics both benefit from the check.
    if field in ("math", "biology"):
        report.gate_reason = f"skipped: field={field} (dimensional check not applicable)"
        return report

    tier_a_findings, in_a, out_a, need_vision = tier_a_check(claims_data, client=client, uploaded_file=uploaded_file)
    report.findings.extend(tier_a_findings)
    report.tier_a_tokens_in = in_a
    report.tier_a_tokens_out = out_a

    for raw in need_vision[:max_vision_calls]:
        finding, in_b, out_b, pages = tier_b_verify(
            doi, pdf_path, raw, claims_data, client=client, uploaded_file=uploaded_file,
        )
        report.tier_b_tokens_in += in_b
        report.tier_b_tokens_out += out_b
        report.tier_b_pages_analyzed += pages
        if finding is not None:
            report.findings.append(finding)

    report.n_findings = len(report.findings)
    report.n_critical = sum(1 for f in report.findings if f.severity == "critical")
    report.n_warning = sum(1 for f in report.findings if f.severity == "warning")
    report.n_info = sum(1 for f in report.findings if f.severity == "info")
    return report


def save_report(report: DimensionalReport, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "dimensional_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
