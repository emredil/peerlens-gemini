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


def _render_math_section(math_json: dict, field_top: str = "physics") -> list[str]:
    lines = []
    if field_top in ("math", "biology"):
        lines.append("## Numerical / symbolic verification")
        lines.append("")
        lines.append(
            "_Not applicable to this paper (field: `" + field_top + "`). "
            "Dimensional and symbolic checks target physics and chemistry equations; "
            "math and biology papers are routed through domain expert and structure agents. "
            "Extracted claims appear in the section above._"
        )
        lines.append("")
        return lines
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


# ---------------------------------------------------------------------------
# New render helpers (target format)
# ---------------------------------------------------------------------------

_SCORE_VERDICT_MAP = [
    (90, "Accept"),
    (70, "Minor revision"),
    (50, "Major revision"),
    (0,  "Reject"),
]


def _score_to_verdict(score: int) -> str:
    for threshold, label in _SCORE_VERDICT_MAP:
        if score >= threshold:
            return label
    return "Reject"


def _render_recommendation_section(report: RefereeReport) -> list[str]:
    lines: list[str] = ["## Recommendation", ""]
    score = int(report.triage_score)
    score_verdict = _score_to_verdict(score)
    domain_verdict = ""
    if report.domain_json:
        domain_verdict = (report.domain_json.get("verdict") or "").strip()

    lines.append(f"**{score_verdict}** — score {score}/100")
    if domain_verdict:
        lines.append(f"**Domain expert verdict:** {domain_verdict}")
    # Prefer domain expert's verdict rationale over the legacy triage reason.
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
    lines: list[str] = ["## Technical details", ""]

    subdomain = ""
    if report.domain_json:
        subdomain = report.domain_json.get("subdomain_used", "") or ""
    lines.append(f"- **Field:** {report.field_top}" + (f", subdomain: {subdomain}" if subdomain else ""))

    claims = report.claims_json or {}
    n_claims = claims.get("n_claims_total") or len(claims.get("claims") or [])
    n_symbols = len(claims.get("symbols") or [])
    lines.append(f"- **Extracted claims:** {n_claims} claim(s), {n_symbols} symbol(s)")

    nov = report.novelty_json or {}
    n_related = nov.get("n_related_retrieved") or 0
    n_corpus = nov.get("n_corpus_matches") or 0
    n_scholar = nov.get("n_scholar_matches") or 0
    lines.append(f"- **Similar papers retrieved:** {n_related} (corpus: {n_corpus}, scholar: {n_scholar})")

    ref_median = nov.get("reference_year_median")
    ref_max = nov.get("reference_year_max")
    if ref_median or ref_max:
        lines.append(f"- **Reference years:** median {ref_median}, most recent {ref_max}")

    math = report.math_json or {}
    n_dim = math.get("n_dimensional_findings") or math.get("dimensional_checks") or 0
    lines.append(f"- **Dimensional checks:** {n_dim} finding(s)")

    verified = math.get("n_verified") or 0
    refuted  = math.get("n_refuted")  or 0
    unable   = math.get("n_unable")   or 0
    lines.append(f"- **Symbolic verification (SymPy):** {verified} verified / {refuted} refuted / {unable} unable")

    g_ver = math.get("n_gemini_verified")
    g_ref = math.get("n_gemini_refuted")
    g_una = math.get("n_gemini_unable")
    if g_ver is not None or g_ref is not None or g_una is not None:
        lines.append(f"- **Symbolic verification (Gemini judgment):** {g_ver or 0} verified / {g_ref or 0} refuted / {g_una or 0} unable")

    agreement = math.get("agreement_sympy_gemini")
    if agreement is not None:
        lines.append(f"- **Agreement between SymPy and Gemini:** {agreement}")

    lines.append(f"- **Model:** gemini-pro-latest")

    # Token totals across agents
    total_in = total_out = 0
    for j in (report.summary_json, report.novelty_json, report.structure_json,
              report.domain_json, report.reproducibility_json, report.claims_json):
        if isinstance(j, dict):
            total_in  += int(j.get("input_tokens") or 0)
            total_out += int(j.get("output_tokens") or 0)
    if total_in or total_out:
        lines.append(f"- **Total tokens:** {total_in} in / {total_out} out")

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# End new helpers
# ---------------------------------------------------------------------------


def render_markdown_LEGACY_UNUSED(report: RefereeReport) -> str:
    """Kept for reference; superseded by render_markdown() above."""
    lines: list[str] = []
    return "\n".join(lines)


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


def _render_reproducibility_section(repro_json: dict, verdict: Verdict) -> list[str]:
    lines = []
    if not repro_json:
        lines.append("## Reproducibility")
        lines.append("")
        lines.append("_No reproducibility analysis was performed._")
        lines.append("")
        return lines

    scrutiny = repro_json.get("scrutiny_level", "unknown")
    lines.append(f"## Reproducibility — verdict: `{verdict}`")
    lines.append("")
    lines.append(f"_Scrutiny level: `{scrutiny}`._")
    lines.append("")
    lines.append(
        f"- Findings: {repro_json.get('n_findings', 0)} "
        f"(crit {repro_json.get('n_critical', 0)}, "
        f"warn {repro_json.get('n_warning', 0)})"
    )
    lines.append("")

    findings = repro_json.get("findings", [])
    if not findings:
        return lines

    kind_label = {
        "missing_code": "Missing code / repository",
        "missing_data": "Missing data availability",
        "missing_params": "Missing parameter specification",
        "missing_algorithm": "Missing algorithm specification",
    }

    by_sev = {"critical": [], "warning": []}
    for f in findings:
        by_sev.setdefault(f.get("severity", "warning"), []).append(f)

    def emit(sev: str, header: str):
        if not by_sev[sev]:
            return
        lines.append(f"**{header}:**")
        lines.append("")
        for f in by_sev[sev]:
            k = kind_label.get(f.get("kind", ""), "Reproducibility gap")
            lines.append(f"- **{k}** at {f.get('location', '')}. "
                         f"{f.get('evidence', '')[:500]}")
            if f.get("suggested_fix"):
                lines.append(f"  _Suggested fix:_ {f['suggested_fix']}")
        lines.append("")

    emit("critical", "Critical")
    emit("warning",  "Requires clarification")
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
