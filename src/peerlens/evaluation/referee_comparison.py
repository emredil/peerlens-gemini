"""Referee comparison (Gemini port): AI referee vs real referee reports."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


@dataclass
class RefereeConcern:
    referee_id: str
    concern: str
    category: str
    severity: str


@dataclass
class ComparisonResult:
    doi_label: str
    ai_findings: list[dict] = field(default_factory=list)
    referee_concerns: list[RefereeConcern] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)
    missed: list[RefereeConcern] = field(default_factory=list)
    novel: list[dict] = field(default_factory=list)
    n_ai_findings: int = 0
    n_referee_math_concerns: int = 0
    n_matches: int = 0
    n_missed: int = 0
    n_novel: int = 0
    summary: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "referee_math_concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "referee_id": {"type": "string"},
                    "concern": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["math_error", "math_clarification", "missed_citation",
                                 "novelty_overreach", "reference_staleness",
                                 "presentation", "reproducibility", "other"],
                    },
                    "severity": {"type": "string", "enum": ["major", "minor", "info"]},
                },
                "required": ["referee_id", "concern", "category", "severity"],
            },
        },
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ai_finding_location": {"type": "string"},
                    "ai_finding_summary": {"type": "string"},
                    "referee_concern_ref": {"type": "string"},
                    "match_quality": {"type": "string", "enum": ["exact", "partial", "adjacent"]},
                    "notes": {"type": "string"},
                },
                "required": ["ai_finding_location", "referee_concern_ref", "match_quality"],
            },
        },
        "missed_concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "referee_id": {"type": "string"},
                    "concern": {"type": "string"},
                    "why_missed": {"type": "string"},
                },
                "required": ["referee_id", "concern"],
            },
        },
        "novel_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ai_finding_location": {"type": "string"},
                    "ai_finding_summary": {"type": "string"},
                    "assessment": {"type": "string", "enum": ["likely_false_positive", "possibly_valuable", "unclear"]},
                    "rationale": {"type": "string"},
                },
                "required": ["ai_finding_location", "assessment"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["referee_math_concerns", "matches", "missed_concerns", "novel_findings", "summary"],
}


SYSTEM_PROMPT = """You are an evaluator comparing an AI-generated referee section
against real referee reports for the same physics paper.

You will be given:
  1. The real referee reports.
  2. Our AI system's report: verdict, findings, and their evidence.

Answer three questions:

  (a) Which concerns did the real referees raise that touch mathematical rigor,
      dimensional consistency, algebraic derivations, equations, limits, or
      symbolic identities; novelty framing, missed citations, prior-art gaps,
      reference staleness; or reproducibility?

  (b) Which of our AI findings correspond to a real concern (match), which
      concerns did we miss (missed), and which of our findings are novel
      (not in any real report)?

  (c) Overall summary in 3-5 sentences.

Guidelines:
  - Be conservative in claiming a match. Two concerns match if they point
    to the same equation, symbol, or reasoning gap.
  - For 'novel_findings', label 'likely_false_positive' if the AI is
    confused about a standard convention. Label 'possibly_valuable' if the
    concern is precise but was overlooked by the humans. 'unclear' otherwise.
  - The 'paraphrase-vs-equation mismatch' AI finding is a system self-check,
    not a paper error — categorise novel findings involving it as
    'likely_false_positive' unless the LaTeX mismatch is real."""


def _load_ai_report(doi: str) -> dict:
    math_path = get_path("processed_data") / "math_reports" / f"{doi}.json"
    nov_path = get_path("processed_data") / "novelty_reports" / f"{doi}.json"

    combined: dict = {"findings": []}
    if math_path.exists():
        m = json.loads(math_path.read_text())
        combined.update({k: v for k, v in m.items() if k != "findings"})
        for f in m.get("findings", []):
            f2 = dict(f)
            f2["origin"] = "math_agent"
            combined["findings"].append(f2)

    if nov_path.exists():
        n = json.loads(nov_path.read_text())
        for f in n.get("findings", []):
            f2 = {
                "origin": "novelty_agent",
                "location": f.get("location", ""),
                "source": "novelty",
                "severity": f.get("severity", ""),
                "verdict": f.get("kind", ""),
                "evidence": f.get("evidence", ""),
                "suggested_fix": f.get("suggested_fix", ""),
                "related_titles": f.get("references", []),
                "paraphrase_mismatch": False,
            }
            combined["findings"].append(f2)
        combined["novelty_related_retrieved"] = n.get("n_related_retrieved", 0)
        combined["novelty_uncited_but_close"] = n.get("n_uncited_but_close", 0)

    str_path = get_path("processed_data") / "structure_reports" / f"{doi}.json"
    if str_path.exists():
        s = json.loads(str_path.read_text())
        for f in s.get("findings", []):
            f2 = {
                "origin": "structure_agent",
                "location": f.get("location", ""),
                "source": "structure",
                "severity": f.get("severity", ""),
                "verdict": f.get("kind", ""),
                "evidence": f.get("evidence", ""),
                "suggested_fix": f.get("suggested_fix", ""),
                "trigger_phrase": f.get("trigger_phrase", ""),
                "paraphrase_mismatch": False,
            }
            combined["findings"].append(f2)
        combined["structure_candidates_scanned"] = s.get("n_candidates_scanned", 0)

    return combined


def _load_referee_reports(doi: str) -> list[dict]:
    path = get_path("raw_data") / "reports" / f"{doi}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    reports = []
    for version, rpt_list in data.get("versions", {}).items():
        for r in rpt_list:
            r["_version"] = version
            reports.append(r)
    return reports


def _format_ai_report(ai: dict) -> str:
    lines = [
        f"AI Referee Report",
        f"  Math verdict:    {ai.get('verdict', 'unknown')}",
        f"  Math reason:     {ai.get('verdict_reason', '')}",
        f"  Dim checks:      {ai.get('n_dimensional_checked', 0)} "
        f"(crit {ai.get('n_dimensional_critical', 0)}, warn {ai.get('n_dimensional_warning', 0)})",
        f"  Symbolic:        {ai.get('n_sympy_tagged', 0)} attempted -> "
        f"verified {ai.get('n_sympy_verified', 0)}, refuted {ai.get('n_sympy_refuted', 0)}, "
        f"unable {ai.get('n_sympy_unable', 0)}, "
        f"paraphrase-mismatch {ai.get('n_paraphrase_mismatch', 0)}",
        f"  Novelty:         related retrieved {ai.get('novelty_related_retrieved', 0)}, "
        f"uncited-but-close {ai.get('novelty_uncited_but_close', 0)}",
        "",
        "Findings (all agents combined):",
    ]
    for i, f in enumerate(ai.get("findings", [])):
        origin = f.get("origin", "?")
        lines.append(
            f"  [{i}] origin={origin}  location={f.get('location', '')}  "
            f"severity={f.get('severity', '')}  verdict={f.get('verdict', '')}"
            + ("  paraphrase_mismatch=True" if f.get("paraphrase_mismatch") else "")
        )
        lines.append(f"       evidence: {f.get('evidence', '')[:400]}")
        if f.get("suggested_fix"):
            lines.append(f"       suggested_fix: {f['suggested_fix']}")
        if f.get("related_titles"):
            for t in f.get("related_titles", [])[:3]:
                lines.append(f"       related: {t[:120]}")
    return "\n".join(lines)


def _format_referee_reports(reports: list[dict]) -> str:
    lines = []
    for r in reports:
        vid = r.get("_version", "?")
        rn = r.get("report_number", "?")
        rid = f"v{vid}_report_{rn}"
        lines.append(f"--- REFEREE {rid} — {r.get('date', '')} — recommendation: {r.get('recommendation', '')} ---")
        if r.get("strengths"):
            lines.append(f"STRENGTHS: {r['strengths'][:600]}")
        if r.get("weaknesses"):
            lines.append(f"WEAKNESSES: {r['weaknesses'][:600]}")
        if r.get("body"):
            lines.append(f"REPORT: {r['body'][:3000]}")
        if r.get("requested_changes"):
            lines.append(f"REQUESTED CHANGES: {r['requested_changes'][:800]}")
        lines.append("")
    return "\n".join(lines)


def compare(doi: str, client: Optional[GeminiClient] = None) -> ComparisonResult:
    if client is None:
        client = GeminiClient()

    ai = _load_ai_report(doi)
    referees = _load_referee_reports(doi)

    result = ComparisonResult(doi_label=doi, ai_findings=ai.get("findings", []))
    if not referees:
        result.summary = "No referee reports available for this paper."
        return result

    user = f"""PAPER: {doi}

=== OUR AI REFEREE REPORT ===

{_format_ai_report(ai)}


=== REAL REFEREE REPORTS ===

{_format_referee_reports(referees)}


Now call the compare_ai_vs_referee tool with your structured comparison."""

    resp = client.structured_call(
        model=CONFIG["gemini"]["model_specialist"],
        system=SYSTEM_PROMPT,
        user=user,
        tool_name="compare_ai_vs_referee",
        tool_description="Structured comparison of AI referee findings vs real referee concerns.",
        tool_input_schema=COMPARISON_SCHEMA,
        max_output_tokens=4096,
        fallback_models=CONFIG["gemini"].get("fallback_chain", []),
    )

    inp = resp.tool_input

    def _ensure_dict(x, default_key: str) -> dict:
        if isinstance(x, dict):
            return x
        return {default_key: str(x)}

    result.referee_concerns = []
    for c in inp.get("referee_math_concerns", []):
        cd = _ensure_dict(c, "concern")
        result.referee_concerns.append(RefereeConcern(
            referee_id=cd.get("referee_id", ""),
            concern=cd.get("concern", str(c)),
            category=cd.get("category", "other"),
            severity=cd.get("severity", "info"),
        ))

    result.matches = [_ensure_dict(m, "ai_finding_summary") for m in inp.get("matches", [])]

    result.missed = []
    for m in inp.get("missed_concerns", []):
        md = _ensure_dict(m, "concern")
        result.missed.append(RefereeConcern(
            referee_id=md.get("referee_id", ""),
            concern=md.get("concern", str(m)),
            category="math_error",
            severity="major",
        ))

    result.novel = [_ensure_dict(n, "ai_finding_summary") for n in inp.get("novel_findings", [])]
    result.summary = inp.get("summary", "")

    result.n_ai_findings = len(result.ai_findings)
    result.n_referee_math_concerns = len(result.referee_concerns)
    result.n_matches = len(result.matches)
    result.n_missed = len(result.missed)
    result.n_novel = len(result.novel)
    result.model = resp.model
    result.input_tokens = resp.input_tokens
    result.output_tokens = resp.output_tokens

    return result


def save_comparison(result: ComparisonResult) -> Path:
    out_dir = get_path("processed_data") / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.doi_label}_comparison.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    return out_path
