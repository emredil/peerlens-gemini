"""Referee Aggregator: unify math + novelty findings into a full AI Referee Section.

Extends the earlier math_aggregator.MathReport with a novelty layer.

Reads:
  - data/processed/claims/{doi}.json
  - data/processed/dimensional_reports/{doi}.json
  - data/processed/sympy_reports/{doi}.json
  - data/processed/novelty_reports/{doi}.json     (NEW)

Produces:
  - data/processed/referee_reports/{doi}.json     (full structured report)
  - data/processed/referee_sections/{doi}_full.md (SciPost-style markdown)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal, Optional

from peerlens.settings import get_path
from peerlens.orchestrator.math_aggregator import aggregate as aggregate_math

import json as _json_stdlib

Verdict = Literal["critical", "major", "minor", "clean", "insufficient_data"]


@dataclass
class RefereeReport:
    doi_label: str
    overall_verdict: Verdict = "insufficient_data"
    overall_reason: str = ""
    math_verdict: Verdict = "insufficient_data"
    novelty_verdict: Verdict = "insufficient_data"
    structure_verdict: Verdict = "insufficient_data"
    domain_verdict: Verdict = "insufficient_data"
    math_json: dict = field(default_factory=dict)
    novelty_json: dict = field(default_factory=dict)
    structure_json: dict = field(default_factory=dict)
    domain_json: dict = field(default_factory=dict)


VERDICT_SEVERITY = {
    "critical": 4,
    "major": 3,
    "minor": 2,
    "clean": 1,
    "insufficient_data": 0,
}

VERDICT_HEADLINES = {
    "critical": "This manuscript has one or more hard errors that must be addressed before publication.",
    "major":    "The manuscript is largely sound, but a small number of concerns require author clarification.",
    "minor":    "The manuscript is consistent as far as automated checks reach; some deeper claims warrant human review.",
    "clean":    "The manuscript passes automated math and novelty checks.",
    "insufficient_data": "Insufficient signal was extracted for a full automated review; a human referee should assess this paper directly.",
}


def _domain_verdict(domain: dict) -> tuple[Verdict, str]:
    """Compute a verdict from a DomainReport dict."""
    if not domain:
        return "insufficient_data", "No domain-expert analysis available."
    findings = domain.get("findings", [])
    if not findings:
        return "clean", "No domain-specific concerns identified."
    counts = {"critical": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f.get("severity", "info")] = counts.get(f.get("severity", "info"), 0) + 1
    if counts["critical"] > 0:
        return "critical", f"{counts['critical']} critical domain concern(s)."
    if counts["warning"] > 0:
        return "major", f"{counts['warning']} domain-specific clarification(s) requested."
    return "minor", "Only informational domain notes recorded."


def _structure_verdict(structure: dict) -> tuple[Verdict, str]:
    """Compute a verdict from a StructureReport dict."""
    if not structure:
        return "insufficient_data", "No structural analysis available."
    findings = structure.get("findings", [])
    if not findings:
        return "clean", "No structural / narrative concerns identified."
    counts = {"critical": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f.get("severity", "info")] = counts.get(f.get("severity", "info"), 0) + 1
    if counts["critical"] > 0:
        return "critical", f"{counts['critical']} critical structural issue(s)."
    if counts["warning"] > 0:
        return "major", f"{counts['warning']} derivation gap(s) or hand-wavy shortcut(s) flagged."
    return "minor", "Only informational structural notes recorded."


def _novelty_verdict(novelty: dict) -> tuple[Verdict, str]:
    """Compute a verdict from a NoveltyReport dict."""
    if not novelty:
        return "insufficient_data", "No novelty analysis available."
    findings = novelty.get("findings", [])
    if not findings:
        return "clean", "No novelty concerns identified."
    counts = {"critical": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f.get("severity", "info")] = counts.get(f.get("severity", "info"), 0) + 1
    if counts["critical"] > 0:
        return "critical", f"{counts['critical']} critical novelty issue(s)."
    if counts["warning"] > 0:
        return "major", f"{counts['warning']} clarification(s) requested on novelty / prior art."
    return "minor", "Only informational novelty notes recorded."


def _combine_verdicts(*verdicts: Verdict) -> Verdict:
    """The overall verdict is the most severe of the components."""
    if not verdicts:
        return "insufficient_data"
    return max(verdicts, key=lambda v: VERDICT_SEVERITY[v])


def aggregate_full(doi_label: str) -> RefereeReport:
    processed = get_path("processed_data")

    # Math
    math_rep = aggregate_math(doi_label)
    math_json = asdict(math_rep)

    # Novelty
    nov_path = processed / "novelty_reports" / f"{doi_label}.json"
    novelty_json: dict = {}
    if nov_path.exists():
        try:
            novelty_json = _json_stdlib.loads(nov_path.read_text())
        except _json_stdlib.JSONDecodeError:
            novelty_json = {}
    novelty_verdict, novelty_reason = _novelty_verdict(novelty_json)

    # Structure
    str_path = processed / "structure_reports" / f"{doi_label}.json"
    structure_json: dict = {}
    if str_path.exists():
        try:
            structure_json = _json_stdlib.loads(str_path.read_text())
        except _json_stdlib.JSONDecodeError:
            structure_json = {}
    structure_verdict, structure_reason = _structure_verdict(structure_json)

    # Domain
    dom_path = processed / "domain_reports" / f"{doi_label}.json"
    domain_json: dict = {}
    if dom_path.exists():
        try:
            domain_json = _json_stdlib.loads(dom_path.read_text())
        except _json_stdlib.JSONDecodeError:
            domain_json = {}
    domain_verdict, domain_reason = _domain_verdict(domain_json)

    overall = _combine_verdicts(math_rep.verdict, novelty_verdict, structure_verdict, domain_verdict)

    if math_rep.verdict == overall:
        overall_reason = f"Math: {math_rep.verdict_reason}"
    elif novelty_verdict == overall:
        overall_reason = f"Novelty: {novelty_reason}"
    elif structure_verdict == overall:
        overall_reason = f"Structure: {structure_reason}"
    else:
        overall_reason = f"Domain: {domain_reason}"

    return RefereeReport(
        doi_label=doi_label,
        overall_verdict=overall,
        overall_reason=overall_reason,
        math_verdict=math_rep.verdict,
        novelty_verdict=novelty_verdict,
        structure_verdict=structure_verdict,
        domain_verdict=domain_verdict,
        math_json=math_json,
        novelty_json=novelty_json,
        structure_json=structure_json,
        domain_json=domain_json,
    )


def save_referee_report(report: RefereeReport) -> Path:
    out_dir = get_path("processed_data") / "referee_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _render_math_section(math_json: dict) -> list[str]:
    """Compact math section — reuse patterns from math_aggregator's renderer."""
    lines = []
    if math_json.get("verdict") == "insufficient_data":
        lines.append("## Mathematical analysis")
        lines.append("")
        lines.append("_Insufficient checkable claims were extracted for a math review._")
        lines.append("")
        return lines

    lines.append(f"## Mathematical analysis — verdict: `{math_json['verdict']}`")
    lines.append("")
    lines.append(f"_{math_json.get('verdict_reason', '')}_")
    lines.append("")
    lines.append(
        f"- Claims extracted: **{math_json.get('n_claims_extracted', 0)}**  "
        f"| Dimensional checks: {math_json.get('n_dimensional_checked', 0)}  "
        f"| Symbolic checks: {math_json.get('n_sympy_tagged', 0)}"
    )
    lines.append(
        f"- Symbolic: verified={math_json.get('n_sympy_verified', 0)}, "
        f"refuted={math_json.get('n_sympy_refuted', 0)}, "
        f"unable={math_json.get('n_sympy_unable', 0)}  "
        f"| Paraphrase mismatches: {math_json.get('n_paraphrase_mismatch', 0)}  "
        f"| Coverage: **{math_json.get('coverage_confidence', 'unknown')}**"
    )
    lines.append("")

    findings = math_json.get("findings", [])
    if not findings:
        return lines

    # dedupe by (source, location)
    seen = set()
    dedup = []
    for f in findings:
        key = (f.get("source", ""), f.get("location", ""), f.get("verdict", ""))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(f)

    # split by severity
    by_sev = {"critical": [], "warning": [], "info": []}
    for f in dedup:
        by_sev.setdefault(f.get("severity", "info"), []).append(f)

    if by_sev["critical"]:
        lines.append("**Critical:**")
        lines.append("")
        for f in by_sev["critical"][:6]:
            lines.append(f"- `{f.get('location', '?')}` [{f.get('source', '?')}] "
                         f"{f.get('verdict', '?')} — {f.get('evidence', '')[:400]}")
        lines.append("")

    if by_sev["warning"]:
        lines.append("**Requires clarification:**")
        lines.append("")
        for f in by_sev["warning"][:8]:
            lines.append(f"- `{f.get('location', '?')}` [{f.get('source', '?')}] "
                         f"{f.get('verdict', '?')} — {f.get('evidence', '')[:400]}")
        lines.append("")

    # note info counts without listing all
    if by_sev["info"]:
        lines.append(f"_Plus {len(by_sev['info'])} informational math checks (see JSON report)._")
        lines.append("")

    return lines


def _render_novelty_section(nov_json: dict, verdict: Verdict, reason: str) -> list[str]:
    lines = []
    if not nov_json:
        lines.append("## Novelty & literature coverage")
        lines.append("")
        lines.append("_No novelty analysis was performed for this manuscript._")
        lines.append("")
        return lines

    lines.append(f"## Novelty & literature coverage — verdict: `{verdict}`")
    lines.append("")
    lines.append(f"_{reason}_")
    lines.append("")
    lines.append(
        f"- Related work retrieved: {nov_json.get('n_related_retrieved', 0)}  "
        f"(corpus: {nov_json.get('n_corpus_matches', 0)}, "
        f"scholar: {nov_json.get('n_scholar_matches', 0)})"
    )
    lines.append(
        f"- Uncited but semantically close: {nov_json.get('n_uncited_but_close', 0)}"
    )
    med_yr = nov_json.get('reference_year_median')
    max_yr = nov_json.get('reference_year_max')
    if med_yr and max_yr:
        lines.append(f"- Reference years: median {med_yr}, most recent {max_yr}")
    lines.append("")

    findings = nov_json.get("findings", [])
    if not findings:
        return lines

    by_sev = {"critical": [], "warning": [], "info": []}
    for f in findings:
        by_sev.setdefault(f.get("severity", "info"), []).append(f)

    kind_label = {
        "missed_citation": "Possibly missed citation",
        "overreach":       "Possible novelty overreach",
        "staleness":       "Reference staleness",
        "originality":     "Originality note",
        "info":            "Note",
    }

    def emit(sev: str, header: str):
        if not by_sev[sev]:
            return
        lines.append(f"**{header}:**")
        lines.append("")
        for f in by_sev[sev]:
            k = kind_label.get(f.get("kind", "info"), "Note")
            lines.append(f"- **{k}** — {f.get('evidence', '')[:500]}")
            if f.get("suggested_fix"):
                lines.append(f"  _Suggested fix:_ {f['suggested_fix']}")
            if f.get("references"):
                for ref in f["references"][:3]:
                    lines.append(f"  · related: {ref[:100]}")
        lines.append("")

    emit("critical", "Critical")
    emit("warning",  "Requires clarification")
    emit("info",     "Informational")
    return lines


def render_markdown(report: RefereeReport) -> str:
    lines: list[str] = []
    lines.append(f"# AI Referee Section — {report.doi_label}")
    lines.append("")
    lines.append(f"**Overall verdict:** `{report.overall_verdict}` — "
                 f"{VERDICT_HEADLINES[report.overall_verdict]}")
    lines.append("")
    lines.append(f"_{report.overall_reason}_")
    lines.append("")
    lines.append(f"- Math verdict:    `{report.math_verdict}`")
    lines.append(f"- Novelty verdict: `{report.novelty_verdict}`")
    lines.append("")

    lines.extend(_render_math_section(report.math_json))
    lines.extend(_render_novelty_section(
        report.novelty_json, report.novelty_verdict,
        report.overall_reason if report.novelty_verdict == report.overall_verdict else "",
    ))
    lines.extend(_render_structure_section(
        report.structure_json, report.structure_verdict,
    ))
    lines.extend(_render_domain_section(
        report.domain_json, report.domain_verdict,
    ))

    lines.append("---")
    lines.append("")
    lines.append("_Generated by PeerLens (prototype). Every finding is an AI suggestion; "
                 "a human editor makes the final decision._")
    return "\n".join(lines)


def _render_structure_section(struct_json: dict, verdict: Verdict) -> list[str]:
    lines = []
    if not struct_json:
        lines.append("## Structure & narrative")
        lines.append("")
        lines.append("_No structural analysis was performed for this manuscript._")
        lines.append("")
        return lines

    lines.append(f"## Structure & narrative — verdict: `{verdict}`")
    lines.append("")
    lines.append(
        f"- Candidates scanned: {struct_json.get('n_candidates_scanned', 0)}  "
        f"| Findings: {struct_json.get('n_findings', 0)} "
        f"(crit {struct_json.get('n_critical', 0)}, "
        f"warn {struct_json.get('n_warning', 0)}, "
        f"info {struct_json.get('n_info', 0)})"
    )
    lines.append("")

    findings = struct_json.get("findings", [])
    if not findings:
        return lines

    by_sev = {"critical": [], "warning": [], "info": []}
    for f in findings:
        by_sev.setdefault(f.get("severity", "info"), []).append(f)

    kind_label = {
        "derivation_gap": "Derivation gap",
        "handwavy_shortcut": "Hand-wavy shortcut",
        "orphan_reference": "Broken / orphan reference",
        "info": "Note",
    }

    def emit(sev: str, header: str):
        if not by_sev[sev]:
            return
        lines.append(f"**{header}:**")
        lines.append("")
        for f in by_sev[sev]:
            k = kind_label.get(f.get("kind", "info"), "Note")
            trigger = f.get("trigger_phrase", "")
            lines.append(f"- **{k}** at {f.get('location', '')}. "
                         f"Trigger: \"{trigger}\"")
            lines.append(f"  {f.get('evidence', '')[:500]}")
            if f.get("suggested_fix"):
                lines.append(f"  _Suggested fix:_ {f['suggested_fix']}")
        lines.append("")

    emit("critical", "Critical")
    emit("warning",  "Requires clarification")
    emit("info",     "Informational")
    return lines


def _render_domain_section(dom_json: dict, verdict: Verdict) -> list[str]:
    lines = []
    if not dom_json:
        lines.append("## Domain expert review")
        lines.append("")
        lines.append("_No domain-specific analysis was performed._")
        lines.append("")
        return lines

    subdomain = dom_json.get("subdomain_used", "unknown")
    persona = dom_json.get("persona", "?")
    lines.append(f"## Domain expert review — verdict: `{verdict}`")
    lines.append("")
    lines.append(f"_Persona applied: {persona} (subdomain: `{subdomain}`)._")
    lines.append("")
    lines.append(
        f"- Findings: {dom_json.get('n_findings', 0)} "
        f"(crit {dom_json.get('n_critical', 0)}, "
        f"warn {dom_json.get('n_warning', 0)}, "
        f"info {dom_json.get('n_info', 0)})"
    )
    lines.append("")

    findings = dom_json.get("findings", [])
    if not findings:
        return lines

    by_sev = {"critical": [], "warning": [], "info": []}
    for f in findings:
        by_sev.setdefault(f.get("severity", "info"), []).append(f)

    kind_label = {
        "domain_concern": "Domain concern",
        "assumption_check": "Assumption check",
        "convention_check": "Convention check",
        "info": "Note",
    }

    def emit(sev: str, header: str):
        if not by_sev[sev]:
            return
        lines.append(f"**{header}:**")
        lines.append("")
        for f in by_sev[sev]:
            k = kind_label.get(f.get("kind", "info"), "Note")
            lines.append(f"- **{k}** at {f.get('location', '')}. "
                         f"{f.get('evidence', '')[:500]}")
            if f.get("suggested_fix"):
                lines.append(f"  _Suggested fix:_ {f['suggested_fix']}")
        lines.append("")

    emit("critical", "Critical")
    emit("warning",  "Requires clarification")
    emit("info",     "Informational")
    return lines


def save_markdown(report: RefereeReport) -> Path:
    out_dir = get_path("processed_data") / "referee_sections"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}_full.md"
    out_path.write_text(render_markdown(report), encoding="utf-8")
    return out_path


def aggregate_and_render_full(doi_label: str) -> tuple[Path, Path]:
    report = aggregate_full(doi_label)
    json_path = save_referee_report(report)
    md_path = save_markdown(report)
    return json_path, md_path
