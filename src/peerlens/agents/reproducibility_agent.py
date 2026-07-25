"""Reproducibility Agent (gate-based).

Design principles:
  1. If the paper is theoretical AND no signals (no code URLs, no
     algorithm mentions, no availability statements) exist in the text,
     skip the Gemini call entirely and return an empty report. Pure
     theory papers get zero reproducibility findings by design.
  2. Only 'warning' and 'critical' severities are allowed. No 'info'.
  3. Only concrete gap kinds are emitted: missing_code, missing_data,
     missing_params, missing_algorithm. No positive_signal, no info.
     If nothing concrete is missing, return an empty findings list.

The goal is that this agent produces zero output unless it has a real,
actionable reproducibility concern that a rigorous referee would raise.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


@dataclass
class ReproFinding:
    kind: str          # "missing_code" | "missing_data" | "missing_params" | "missing_algorithm"
    severity: str      # "critical" | "warning"
    location: str
    evidence: str
    suggested_fix: str = ""
    confidence: float = 0.0


@dataclass
class ReproReport:
    doi_label: str
    paper_types: list[str] = field(default_factory=list)
    scrutiny_level: str = "none"  # none | medium | deep
    detected_signals: dict = field(default_factory=dict)
    gate_reason: str = ""
    n_findings: int = 0
    n_critical: int = 0
    n_warning: int = 0
    findings: list[ReproFinding] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


CODE_URL_PATTERNS = [
    re.compile(r"https?://(?:www\.)?github\.com/[\w\-/.]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?gitlab\.com/[\w\-/.]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?bitbucket\.org/[\w\-/.]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?zenodo\.org/[\w\-/.]+", re.IGNORECASE),
    re.compile(r"https?://doi\.org/10\.5281/[\w.-]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?figshare\.com/[\w\-/.]+", re.IGNORECASE),
]

AVAILABILITY_KEYWORDS = [
    "code available", "code is available", "source code",
    "data available", "data is available", "data availability",
    "reproducibility statement", "computational reproducibility",
    "supplementary code", "publicly available", "open-source",
]

ALGORITHM_KEYWORDS = [
    "monte carlo", "MCMC", "tensor network", "DMRG", "MPS",
    "matrix product", "quantum monte carlo",
    "molecular dynamics", "lattice qcd", "finite element",
    "runge-kutta", "verlet", "exact diagonalization",
    "hartree-fock", "density functional", "DFT",
    "path integral", "worldline", "lattice regularization",
    "numerical simulation", "numerical solver",
]


def _scan_signals(parsed: dict) -> dict:
    text_parts = [parsed.get("title", ""), parsed.get("abstract", "")]
    for s in parsed.get("sections", []):
        text_parts.append(s.get("text", ""))
    for r in parsed.get("references", []):
        text_parts.append(r)
    full_text = "\n".join(text_parts)
    full_text_lower = full_text.lower()

    urls = []
    for pat in CODE_URL_PATTERNS:
        for m in pat.finditer(full_text):
            url = m.group(0)
            if url not in urls:
                urls.append(url)

    availability = [kw for kw in AVAILABILITY_KEYWORDS if kw in full_text_lower]
    algorithms = [kw for kw in ALGORITHM_KEYWORDS if kw.lower() in full_text_lower]

    return {
        "code_urls": urls,
        "availability_statements": availability,
        "algorithms_mentioned": algorithms,
        "n_tables": len(parsed.get("table_captions", [])),
        "n_figures": len(parsed.get("figure_captions", [])),
    }


def _classification_for(doi: str) -> Optional[dict]:
    path = get_path("processed_data") / "classifications" / f"{doi}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _decide_gate(paper_types: list[str], signals: dict) -> tuple[str, str]:
    """Return (scrutiny_level, gate_reason).

    Levels:
      none   : skip Gemini entirely, empty report
      medium : some signals present but paper is nominally theoretical
      deep   : paper_type explicitly computational / experimental /
               phenomenological, or multiple strong signals present
    """
    heavy = {"computational", "experimental", "phenomenological"}
    is_heavy = bool(paper_types) and any(t in heavy for t in paper_types)

    n_urls = len(signals.get("code_urls", []))
    n_avail = len(signals.get("availability_statements", []))
    n_algo = len(signals.get("algorithms_mentioned", []))

    strong = 0
    if n_urls > 0: strong += 1
    if n_avail > 0: strong += 1
    if n_algo >= 2: strong += 1

    if is_heavy or strong >= 2:
        return "deep", f"heavy paper_type={is_heavy}, strong_signals={strong}"

    if strong >= 1:
        return "medium", f"1 strong signal (urls={n_urls}, avail={n_avail}, algos={n_algo})"

    if n_algo >= 1:
        return "medium", f"single algorithm mention: {signals.get('algorithms_mentioned')}"

    return "none", "no signals detected, paper appears purely theoretical"


REPRO_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["missing_code", "missing_data",
                                 "missing_params", "missing_algorithm"],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "warning"],
                    },
                    "location": {"type": "string"},
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["kind", "severity", "location", "evidence", "confidence"],
            },
        },
    },
    "required": ["findings"],
}


SYSTEM_TEMPLATES = {
    "deep": """You are a physics referee focused on REPRODUCIBILITY of a computational,
experimental, or phenomenological paper.

Emit findings ONLY for CONCRETE, ACTIONABLE reproducibility GAPS. A gap is
worth flagging only if a rigorous referee would ask the author to add
specific information before publication.

Allowed finding kinds (no others exist):
  - missing_code:      Numerical or computational results are shown but
                       no code repository is cited or made available.
  - missing_data:      Data used to generate figures / tables is not
                       accompanied by a data availability statement.
  - missing_params:    A simulation or numerical calculation is described
                       but specific parameter values (system size,
                       coupling constants, seeds, iteration counts,
                       precision) are not fully reported.
  - missing_algorithm: A method is invoked (Monte Carlo, DMRG, exact
                       diagonalization, DFT, ...) without specifying
                       which library, variant, or version.

Severity: 'warning' unless the paper's central result rests on the
un-shared / unspecified item, in which case 'critical'.

STRICT RULES:
  1. Do NOT emit findings for good practices, positive observations,
     acknowledgments, or descriptive notes. If the paper does something
     well, stay silent — positive notes are not your job.
  2. Do NOT emit findings that only paraphrase what the paper already
     states elsewhere.
  3. If NOTHING concrete is missing, return an EMPTY findings list.
     Empty is the correct answer whenever there is no gap.

Target: 0 to 4 findings. Silence is preferable to filler.""",

    "medium": """You are a physics referee checking reproducibility. The paper is
nominally theoretical but has some computational or numerical component.

Emit findings ONLY for concrete missing information a rigorous referee
would demand before publication:
  - missing_code:      Numerics shown, code not shared.
  - missing_params:    Specific numerical values / plots without the
                       parameters that generated them.

Allowed severities: 'warning' or 'critical' only.

STRICT RULES:
  1. Do NOT emit findings for good practices or acknowledgments.
  2. Do NOT emit findings that duplicate paper content.
  3. Empty findings list is the correct answer whenever nothing
     concrete is missing.

Target: 0 to 2 findings. Silence is the default.""",
}


def _format_paper_context(parsed: dict, signals: dict) -> str:
    section_titles = ", ".join(
        f"{s.get('number','?')} {s.get('title','?')}"
        for s in parsed.get("sections", [])[:15]
    )
    return f"""TITLE:
{parsed.get('title', '')}

ABSTRACT:
{parsed.get('abstract', '')}

SECTION HEADINGS:
{section_titles}

REPRODUCIBILITY SIGNALS DETECTED (from pattern scan):
  Code / data URLs found: {signals['code_urls'] or 'NONE'}
  Availability keyword hits: {signals['availability_statements'] or 'NONE'}
  Algorithm mentions: {signals['algorithms_mentioned'] or 'NONE'}
  Tables: {signals['n_tables']}, Figures: {signals['n_figures']}"""


def analyze_reproducibility(doi: str, client: Optional[GeminiClient] = None) -> ReproReport:
    report = ReproReport(doi_label=doi)

    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not parsed_path.exists():
        report.gate_reason = "parsed JSON missing"
        return report
    parsed = json.loads(parsed_path.read_text())

    classification = _classification_for(doi)
    paper_types = (classification or {}).get("paper_types", []) or []
    report.paper_types = paper_types

    signals = _scan_signals(parsed)
    report.detected_signals = signals

    scrutiny, reason = _decide_gate(paper_types, signals)
    report.scrutiny_level = scrutiny
    report.gate_reason = reason

    # GATE: if no scrutiny needed, skip Gemini entirely
    if scrutiny == "none":
        return report

    if client is None:
        client = GeminiClient()

    system_prompt = SYSTEM_TEMPLATES[scrutiny]
    user = f"""PAPER CONTEXT:

{_format_paper_context(parsed, signals)}


Emit findings ONLY for concrete gaps. Empty findings list is the
correct answer whenever nothing concrete is missing."""

    result = client.structured_call(
        model=CONFIG["gemini"]["model_specialist"],
        system=system_prompt,
        user=user,
        tool_name="record_reproducibility_findings",
        tool_description="Record reproducibility referee findings (warning/critical only).",
        tool_input_schema=REPRO_SCHEMA,
        max_output_tokens=2500,
        fallback_models=CONFIG["gemini"].get("fallback_chain", []),
    )

    for f in result.tool_input.get("findings", []):
        # Defensive: schema forbids info but be safe
        if f.get("severity") not in ("critical", "warning"):
            continue
        if f.get("kind") not in ("missing_code", "missing_data",
                                  "missing_params", "missing_algorithm"):
            continue
        report.findings.append(ReproFinding(
            kind=f["kind"],
            severity=f["severity"],
            location=f.get("location", ""),
            evidence=f["evidence"],
            suggested_fix=f.get("suggested_fix", ""),
            confidence=f.get("confidence", 0.0),
        ))

    report.n_findings = len(report.findings)
    report.n_critical = sum(1 for f in report.findings if f.severity == "critical")
    report.n_warning = sum(1 for f in report.findings if f.severity == "warning")
    report.model = result.model
    report.input_tokens = result.input_tokens
    report.output_tokens = result.output_tokens
    return report


def save_report(report: ReproReport, out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "reproducibility_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
