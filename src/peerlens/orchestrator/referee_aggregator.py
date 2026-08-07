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

from peerlens.orchestrator.claim_render import render_claims_section
from peerlens.settings import get_path
from peerlens.orchestrator.math_aggregator import aggregate as aggregate_math
from peerlens.orchestrator.scoring import score_paper, save_triage_score

import json as _json_stdlib

Verdict = Literal["critical", "major", "minor", "clean", "insufficient_data"]


@dataclass
class RefereeReport:
    doi_label: str
    overall_verdict: Verdict = "insufficient_data"
    overall_reason: str = ""
    triage_score: int = 100
    triage_verdict: str = "insufficient_data"
    triage_top_negative: list = field(default_factory=list)
    math_verdict: Verdict = "insufficient_data"
    novelty_verdict: Verdict = "insufficient_data"
    structure_verdict: Verdict = "insufficient_data"
    domain_verdict: Verdict = "insufficient_data"
    reproducibility_verdict: Verdict = "insufficient_data"
    math_json: dict = field(default_factory=dict)
    novelty_json: dict = field(default_factory=dict)
    structure_json: dict = field(default_factory=dict)
    domain_json: dict = field(default_factory=dict)
    reproducibility_json: dict = field(default_factory=dict)
    summary_json: dict = field(default_factory=dict)
    policy_json: dict = field(default_factory=dict)
    claims_json: dict = field(default_factory=dict)
    field_top: str = "physics"
    title: str = ""
    n_pages: int = 0 


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


def _reproducibility_verdict(repro: dict) -> tuple[Verdict, str]:
    if not repro:
        return "insufficient_data", "No reproducibility analysis available."
    findings = repro.get("findings", [])
    if not findings:
        return "clean", "No reproducibility concerns identified."
    counts = {"critical": 0, "warning": 0}
    for f in findings:
        counts[f.get("severity", "warning")] = counts.get(f.get("severity", "warning"), 0) + 1
    if counts["critical"] > 0:
        return "critical", f"{counts['critical']} critical reproducibility issue(s)."
    return "major", f"{counts['warning']} reproducibility clarification(s) requested."


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

    repro_path = processed / "reproducibility_reports" / f"{doi_label}.json"
    repro_json: dict = {}
    if repro_path.exists():
        try:
            repro_json = _json_stdlib.loads(repro_path.read_text())
        except _json_stdlib.JSONDecodeError:
            repro_json = {}
    repro_verdict, repro_reason = _reproducibility_verdict(repro_json)

    # Summary (plain-language narrative for the editor). Optional; only
    # renders when the summary agent has produced a report on disk.
    sum_path = processed / "summary_reports" / f"{doi_label}.json"
    summary_json: dict = {}
    if sum_path.exists():
        try:
            summary_json = _json_stdlib.loads(sum_path.read_text())
        except _json_stdlib.JSONDecodeError:
            summary_json = {}

    # Optional policy check (LLM-authorship). Independent of triage score;
    # renders only when the editor opted in for this paper.
    policy_path = processed / "llm_detection_reports" / f"{doi_label}.json"
    policy_json: dict = {}
    if policy_path.exists():
        try:
            policy_json = _json_stdlib.loads(policy_path.read_text())
        except _json_stdlib.JSONDecodeError:
            policy_json = {}

    parsed_path = processed / "parsed" / f"{doi_label}.json"
    parsed_meta: dict = {}
    if parsed_path.exists():
        try:
            parsed_meta = _json_stdlib.loads(parsed_path.read_text())
        except _json_stdlib.JSONDecodeError:
            parsed_meta = {}
    title_str = parsed_meta.get("title", "") or ""
    n_pages_int = int(parsed_meta.get("n_pages", 0) or 0)

    cls_path = processed / "classifications" / f"{doi_label}.json"
    cls_json: dict = {}
    if cls_path.exists():
        try:
            cls_json = _json_stdlib.loads(cls_path.read_text())
        except _json_stdlib.JSONDecodeError:
            cls_json = {}
    field_top = cls_json.get("field_top", "physics")

    claims_path = processed / "claims" / f"{doi_label}.json"
    claims_json: dict = {}
    if claims_path.exists():
        try:
            claims_json = _json_stdlib.loads(claims_path.read_text())
        except _json_stdlib.JSONDecodeError:
            claims_json = {}

    overall = _combine_verdicts(math_rep.verdict, novelty_verdict,
                                structure_verdict, domain_verdict, repro_verdict)

    if math_rep.verdict == overall:
        overall_reason = f"Math: {math_rep.verdict_reason}"
    elif novelty_verdict == overall:
        overall_reason = f"Novelty: {novelty_reason}"
    elif structure_verdict == overall:
        overall_reason = f"Structure: {structure_reason}"
    elif domain_verdict == overall:
        overall_reason = f"Domain: {domain_reason}"
    else:
        overall_reason = f"Reproducibility: {repro_reason}"

    triage = score_paper(doi_label)
    save_triage_score(triage)
    top_neg_serialized = [
        {
            "source": c.source,
            "kind": c.kind,
            "severity": c.severity,
            "location": c.location,
            "weight": c.weight,
            "evidence_short": c.evidence_short,
        } for c in triage.top_negative
    ]

    return RefereeReport(
        doi_label=doi_label,
        overall_verdict=overall,
        overall_reason=overall_reason,
        math_verdict=math_rep.verdict,
        novelty_verdict=novelty_verdict,
        structure_verdict=structure_verdict,
        domain_verdict=domain_verdict,
        reproducibility_verdict=repro_verdict,
        triage_score=triage.score,
        triage_verdict=triage.verdict,
        triage_top_negative=top_neg_serialized,
        math_json=math_json,
        novelty_json=novelty_json,
        structure_json=structure_json,
        domain_json=domain_json,
        reproducibility_json=repro_json,
        summary_json=summary_json,
        policy_json=policy_json,
        claims_json=claims_json,
        field_top=field_top,
        title=title_str,
        n_pages=n_pages_int,
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
def render_markdown(report: RefereeReport) -> str:
    """Referee-style markdown report matching the target editor format."""
    lines: list[str] = []

    # Header
    title = report.title or report.doi_label
    lines.append(f"# PeerLens Referee Report — {title}")
    meta_bits = [f"`{report.doi_label}`"]
    if report.n_pages:
        meta_bits.append(f"{report.n_pages} pages")
    if report.field_top:
        meta_bits.append(report.field_top)
    lines.append("_" + " · ".join(meta_bits) + "_")
    lines.append("")

    # Recommendation
    lines.extend(_render_recommendation_section(report))

    # Summary (from summary agent)
    if report.summary_json:
        lines.extend(_render_summary_section(report.summary_json))

    # Strengths + Weaknesses (from domain expert)
    lines.extend(_render_strengths_section(report.domain_json))
    lines.extend(_render_weaknesses_section(report.domain_json))

    # Section-by-section commentary
    lines.extend(_render_section_by_section(report))

    # Minor comments (warning/info findings across agents)
    lines.extend(_render_minor_comments_section(report))

    # Policy (opt-in, LLM authorship)
    if report.policy_json:
        lines.extend(_render_policy_section(report.policy_json))

    lines.append("---")
    lines.append("")

    # Technical details (metrics + tokens)
    lines.extend(_render_technical_details_section(report))

    lines.append("---")
    lines.append("")
    lines.append("_Generated by PeerLens (prototype). Every finding is an AI suggestion; "
                 "a human editor makes the final decision._")
    return "\n".join(lines)
def _render_recommendation_section(report: RefereeReport) -> list[str]:
    """Domain expert's verdict + rationale. No numeric score.

    The legacy triage_score is intentionally not rendered: the numeric
    score never gained editorial meaning (it consistently maxed out at
    100/100 even for papers with critical math findings), and the
    domain-expert verdict combined with a 1-2 sentence rationale is
    what an editor actually acts on.
    """
    lines: list[str] = ["## Recommendation", ""]
    domain_verdict = ""
    if report.domain_json:
        domain_verdict = (report.domain_json.get("verdict") or "").strip()
    if not domain_verdict:
        domain_verdict = "insufficient data"
    # Capitalise first letter for the header line, keep rest as written.
    display = domain_verdict[:1].upper() + domain_verdict[1:]
    lines.append(f"**{display}**")

    rationale = ""
    if report.domain_json:
        rationale = (report.domain_json.get("verdict_rationale") or "").strip()
    if not rationale:
        rationale = (report.overall_reason or "").strip()
    if rationale:
        lines.append("")
        lines.append(f"_{rationale}_")
    lines.append("")
    return lines


def _render_strengths_section(domain_json: dict) -> list[str]:
    lines: list[str] = []
    if not domain_json:
        return lines
    strengths = (domain_json.get("strengths_narrative") or "").strip()
    if not strengths:
        return lines
    lines.append("## Strengths")
    lines.append("")
    lines.append(strengths)
    lines.append("")
    return lines


def _render_weaknesses_section(domain_json: dict) -> list[str]:
    lines: list[str] = ["## Weaknesses", ""]
    weaknesses = ""
    if domain_json:
        weaknesses = (domain_json.get("weaknesses_narrative") or "").strip()
    if weaknesses:
        lines.append(weaknesses)
    else:
        lines.append("No major weaknesses identified.")
    lines.append("")
    return lines


def _render_section_by_section(report: RefereeReport) -> list[str]:
    lines: list[str] = ["## Section-by-section commentary", ""]

    nov_narrative = (report.novelty_json.get("narrative") or "").strip() if report.novelty_json else ""
    if nov_narrative:
        lines.append("### On novelty and prior work")
        lines.append("")
        lines.append(nov_narrative)
        lines.append("")

    dom_comment = (report.domain_json.get("commentary_narrative") or "").strip() if report.domain_json else ""
    if dom_comment:
        subdomain = report.domain_json.get("subdomain_used", "") or "domain-specific"
        lines.append(f"### On {subdomain} content")
        lines.append("")
        lines.append(dom_comment)
        lines.append("")

    struct_narrative = (report.structure_json.get("narrative") or "").strip() if report.structure_json else ""
    if struct_narrative:
        lines.append("### On presentation and clarity")
        lines.append("")
        lines.append(struct_narrative)
        lines.append("")

    repro_narrative = (report.reproducibility_json.get("narrative") or "").strip() if report.reproducibility_json else ""
    repro_verdict = (report.reproducibility_json.get("verdict") or "").strip().lower() if report.reproducibility_json else ""
    if repro_narrative and repro_verdict != "not applicable":
        lines.append("### On reproducibility")
        lines.append("")
        lines.append(repro_narrative)
        lines.append("")

    math_prose = _render_math_prose(report.math_json)
    if math_prose:
        lines.append("### On numerical and symbolic consistency")
        lines.append("")
        lines.extend(math_prose)
        lines.append("")

    # If nothing rendered under section-by-section, drop the header too.
    if len(lines) == 2:
        return []
    return lines


def _render_math_prose(math_json: dict) -> list[str]:
    """Turn structured math findings into a short prose paragraph."""
    if not math_json:
        return []
    verdict = (math_json.get("verdict") or "").strip()
    findings = math_json.get("findings") or []
    n_findings = len(findings) if isinstance(findings, list) else 0

    if verdict in ("clean", "insufficient_data") and n_findings == 0:
        return []

    sentences: list[str] = []
    critical = [f for f in findings if isinstance(f, dict) and f.get("severity") == "critical"]
    warning  = [f for f in findings if isinstance(f, dict) and f.get("severity") in ("major", "warning")]

    if critical:
        sentences.append(f"Automated checks flagged {len(critical)} critical algebraic or dimensional issue(s) that a human referee should review.")
        for f in critical[:3]:
            loc = f.get("location") or "unspecified location"
            ev = (f.get("evidence_short") or f.get("evidence") or "").strip()
            if ev:
                sentences.append(f"At {loc}: {ev}")
    if warning:
        sentences.append(f"An additional {len(warning)} warning-level finding(s) were noted; these are unlikely to affect the paper's conclusions but may benefit from clarification.")

    if not sentences:
        sentences.append("Automated dimensional and symbolic checks were performed on the paper's central claims; no major inconsistencies were flagged.")

    return sentences


def _render_minor_comments_section(report: RefereeReport) -> list[str]:
    """Collect warning/info-level findings from every agent."""
    minor_items: list[str] = []

    for label, agent_json in [
        ("Novelty", report.novelty_json),
        ("Structure", report.structure_json),
        ("Domain", report.domain_json),
        ("Reproducibility", report.reproducibility_json),
    ]:
        if not agent_json:
            continue
        findings = agent_json.get("findings") or []
        for f in findings if isinstance(findings, list) else []:
            if not isinstance(f, dict):
                continue
            sev = (f.get("severity") or "").lower()
            if sev in ("minor", "warning", "info"):
                loc = f.get("location") or ""
                ev = (f.get("evidence_short") or f.get("evidence") or f.get("message") or "").strip()
                if ev:
                    tag = f"[{label}]"
                    if loc:
                        minor_items.append(f"- {tag} {loc}: {ev}")
                    else:
                        minor_items.append(f"- {tag} {ev}")

    if not minor_items:
        return []
    lines: list[str] = ["## Minor comments", ""]
    lines.extend(minor_items)
    lines.append("")
    return lines


def _render_technical_details_section(report: RefereeReport) -> list[str]:
    """Minimal metadata footer: model, field, reference year median."""
    lines: list[str] = ["## Technical details", ""]

    lines.append("- **Model:** gemini-pro-latest")

    subdomain = ""
    if report.domain_json:
        subdomain = report.domain_json.get("subdomain_used", "") or ""
    lines.append(f"- **Field:** {report.field_top}" + (f", subdomain: {subdomain}" if subdomain else ""))

    lines.append("")
    return lines
def _render_summary_section(summary_json: dict) -> list[str]:
    """Render the plain-language summary as a top-of-report section."""
    lines: list[str] = []
    narrative = (summary_json.get("narrative") or "").strip()
    if not narrative:
        return lines
    lines.append("## Summary")
    lines.append("")
    lines.append(narrative)
    lines.append("")
    return lines
def _render_policy_section(policy_json: dict) -> list[str]:
    """Render the opt-in LLM-authorship policy check.

    This section is a policy signal, not a quality signal, and does not
    influence the triage score. It appears only when the editor explicitly
    requested it for this paper (via --check-llm or the web UI checkbox).
    """
    lines: list[str] = []
    if not policy_json:
        return lines

    verdict = policy_json.get("verdict", "unclear")
    confidence = policy_json.get("confidence", 0.0)
    reasoning = policy_json.get("reasoning", "")
    signals = policy_json.get("signals", []) or []
    model = policy_json.get("model", "")

    label = str(verdict).upper().replace("_", " ")
    try:
        conf_txt = f"{float(confidence):.2f}"
    except (TypeError, ValueError):
        conf_txt = "?"

    lines.append(f"## Policy check — {label} (confidence {conf_txt})")
    lines.append("")
    lines.append("_Opt-in AI-authorship detection. Does not affect the triage score._")
    if model:
        lines.append(f"_Model: `{model}`._")
    lines.append("")

    if reasoning:
        lines.append(f"**Reasoning.** {reasoning}")
        lines.append("")

    if signals:
        n_strong = sum(1 for s in signals if s.get("severity") == "strong")
        n_moderate = sum(1 for s in signals if s.get("severity") == "moderate")
        n_weak = sum(1 for s in signals if s.get("severity") == "weak")
        lines.append(
            f"**Signals:** {len(signals)} total "
            f"(strong={n_strong}, moderate={n_moderate}, weak={n_weak})"
        )
        lines.append("")
        for s in signals:
            sev = s.get("severity", "?")
            cat = s.get("category", "?")
            loc = s.get("location", "") or "—"
            desc = s.get("description", "")
            lines.append(f"- **[{sev}]** _{cat}_ @ {loc}: {desc}")
        lines.append("")

    lines.append("---")
    lines.append("")
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
