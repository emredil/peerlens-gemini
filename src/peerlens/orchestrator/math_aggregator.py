"""Math Aggregator: synthesize dimensional + SymPy findings into a single
'Math Referee Section' per paper.

Reads:
  - data/processed/claims/{doi}.json          (extractor output, for coverage)
  - data/processed/dimensional_reports/{doi}.json
  - data/processed/sympy_reports/{doi}.json

Produces:
  - data/processed/math_reports/{doi}.json     (structured)
  - data/processed/referee_sections/{doi}_math.md  (human-readable, SciPost tone)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal, Optional

from peerlens.settings import get_path


Verdict = Literal["critical", "major", "minor", "clean", "insufficient_data"]


@dataclass
class AggregatedFinding:
    source: str          # "dimensional" | "sympy" | "extractor_meta"
    location: str
    verdict: str         # native verdict from the source
    severity: str        # critical | warning | info
    evidence: str
    suggested_fix: str = ""
    confidence: float = 0.0
    paraphrase_mismatch: bool = False   # flagged specially in review section


@dataclass
class MathReport:
    doi_label: str
    verdict: Verdict = "insufficient_data"
    verdict_reason: str = ""

    n_claims_extracted: int = 0
    n_sympy_tagged: int = 0
    n_sympy_verified: int = 0
    n_sympy_refuted: int = 0
    n_sympy_unable: int = 0
    n_dimensional_checked: int = 0
    n_dimensional_critical: int = 0
    n_dimensional_warning: int = 0
    n_paraphrase_mismatch: int = 0

    coverage_confidence: str = "unknown"  # high | medium | low
    findings: list[AggregatedFinding] = field(default_factory=list)


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _compute_verdict(report: MathReport) -> tuple[Verdict, str]:
    """Rules for the top-line verdict.

    Key nuance: a `refuted` sympy finding that is ALSO flagged as a
    paraphrase-vs-equation mismatch is not a paper-level error — it is a
    disagreement between the AI's own paraphrase and the source equation.
    We subtract these from the 'genuine refuted' count before deciding
    severity, so the top-line verdict tracks the paper's math, not the
    system's self-consistency.
    """
    genuine_refuted = max(report.n_sympy_refuted - report.n_paraphrase_mismatch, 0)

    if report.n_dimensional_critical > 0:
        return "critical", (
            f"{report.n_dimensional_critical} dimensional inconsistency(ies) found — "
            "these are hard errors."
        )
    if genuine_refuted > 0:
        return "critical", (
            f"{genuine_refuted} symbolic claim(s) failed verification against the "
            "source equations. Equation-vs-claim contradictions were detected."
        )
    if report.n_dimensional_warning > 0:
        return "major", (
            f"{report.n_dimensional_warning} dimensional clarification request(s) "
            "the author should address."
        )
    if report.n_paraphrase_mismatch > 0:
        return "minor", (
            f"{report.n_paraphrase_mismatch} paraphrase-vs-equation mismatch(es) "
            "flagged. These likely indicate imperfect summarisation by the AI "
            "rather than a real error in the paper. A human referee should confirm."
        )
    if report.n_sympy_unable > report.n_sympy_verified and report.n_sympy_tagged >= 5:
        return "minor", (
            f"Coverage limited: {report.n_sympy_unable}/{report.n_sympy_tagged} sympy-tagged claims "
            "could not be verified symbolically (operator-algebra limits). Human reviewer "
            "should validate these manually."
        )
    if report.n_sympy_verified > 0 or report.n_dimensional_checked > 0:
        return "clean", "No dimensional or symbolic issues detected in checked claims."
    return "insufficient_data", "Not enough checkable claims to form a verdict."


def _coverage_confidence(report: MathReport) -> str:
    """How much of the paper's math did we actually test?"""
    tested = report.n_sympy_verified + report.n_sympy_refuted + report.n_dimensional_checked
    total = report.n_claims_extracted
    if total == 0:
        return "unknown"
    ratio = tested / total
    if ratio >= 0.5:
        return "high"
    if ratio >= 0.25:
        return "medium"
    return "low"


def aggregate(doi_label: str) -> MathReport:
    """Build a MathReport by joining the three per-agent JSONs."""
    processed = get_path("processed_data")
    claims = _load_json(processed / "claims" / f"{doi_label}.json") or {}
    dim = _load_json(processed / "dimensional_reports" / f"{doi_label}.json") or {}
    sym = _load_json(processed / "sympy_reports" / f"{doi_label}.json") or {}

    report = MathReport(doi_label=doi_label)

    # Extractor coverage stats
    report.n_claims_extracted = len(claims.get("claims", []))
    report.n_sympy_tagged = sum(
        1 for c in claims.get("claims", []) if c.get("verification_method") == "sympy"
    )

    # Ingest dimensional findings
    for f in dim.get("findings", []):
        sev = f.get("severity", "info")
        aggr = AggregatedFinding(
            source="dimensional",
            location=f.get("location", ""),
            verdict=f.get("verdict", ""),
            severity=sev,
            evidence=f.get("evidence", ""),
            suggested_fix=f.get("suggested_fix", ""),
            confidence=f.get("confidence", 0.0),
        )
        report.findings.append(aggr)
        if sev == "critical":
            report.n_dimensional_critical += 1
        elif sev == "warning":
            report.n_dimensional_warning += 1
    report.n_dimensional_checked = len(dim.get("findings", []))

    # Ingest sympy findings
    for f in sym.get("findings", []):
        v = f.get("verdict", "")
        sev = f.get("severity", "info")
        detail = f.get("detail", "") or ""
        mismatch = "paraphrase-latex mismatch" in detail.lower()
        aggr = AggregatedFinding(
            source="sympy",
            location=f.get("location", ""),
            verdict=v,
            severity=sev,
            evidence=detail[:800],
            suggested_fix="",
            confidence=0.0,
            paraphrase_mismatch=mismatch,
        )
        report.findings.append(aggr)
        if v == "verified":
            report.n_sympy_verified += 1
        elif v == "refuted":
            report.n_sympy_refuted += 1
        elif v == "unable_to_verify":
            report.n_sympy_unable += 1
        if mismatch:
            report.n_paraphrase_mismatch += 1

    # Compute top-line
    report.verdict, report.verdict_reason = _compute_verdict(report)
    report.coverage_confidence = _coverage_confidence(report)

    return report


def save_math_report(report: MathReport) -> Path:
    out_dir = get_path("processed_data") / "math_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# Markdown renderer — SciPost referee section format
# ---------------------------------------------------------------------------

VERDICT_HEADLINES = {
    "critical": "This paper has one or more hard mathematical errors that must be addressed before publication.",
    "major":    "The mathematics is largely sound, but a small number of claims require author clarification.",
    "minor":    "Analysis is consistent as far as automated checking reaches; some deeper claims remain unverified and warrant human review.",
    "clean":    "The mathematical content passes both dimensional and symbolic consistency checks.",
    "insufficient_data": "Insufficient checkable claims were extracted from this paper; a purely automated math review is not applicable.",
}


def render_markdown(report: MathReport) -> str:
    """Render a de-duplicated, readable Math Referee Section.

    Grouping strategy:
      - Findings are grouped by location. Within a location, the most severe
        finding becomes the primary bullet; other findings on the same
        location are shown as compact cross-checks below it.
      - Paraphrase mismatches get their own top block so the editor can
        distinguish 'AI misread the paper' from 'the paper is wrong'.
      - Findings already listed in the mismatch block are NOT repeated in the
        critical/warning blocks.
    """
    lines: list[str] = []
    lines.append(f"# Math Analysis — {report.doi_label}")
    lines.append("")
    lines.append(f"**Verdict:** `{report.verdict}` — {VERDICT_HEADLINES[report.verdict]}")
    lines.append("")
    lines.append(f"_Reason: {report.verdict_reason}_")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Claims extracted from paper: **{report.n_claims_extracted}**")
    lines.append(f"- Dimensional checks performed: {report.n_dimensional_checked} "
                 f"(critical: {report.n_dimensional_critical}, warning: {report.n_dimensional_warning})")
    lines.append(f"- Symbolic checks: {report.n_sympy_tagged} attempted → "
                 f"verified: {report.n_sympy_verified}, refuted: {report.n_sympy_refuted}, "
                 f"unable_to_verify: {report.n_sympy_unable}")
    lines.append(f"- Paraphrase↔equation mismatches flagged: {report.n_paraphrase_mismatch}")
    lines.append(f"- Coverage confidence: **{report.coverage_confidence}**")
    lines.append("")

    # ---- group findings by location, deduplicate ----
    _SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}

    by_location: dict[str, list[AggregatedFinding]] = {}
    for f in report.findings:
        by_location.setdefault(f.location or "(unlocated)", []).append(f)

    # Within each location, sort by severity
    for loc in by_location:
        by_location[loc].sort(key=lambda f: (_SEV_ORDER.get(f.severity, 3), f.source))

    # Locations that appear as paraphrase mismatches — handled in their own block
    mismatch_locations = {
        f.location for f in report.findings if f.paraphrase_mismatch
    }

    # ---- paraphrase mismatch block ----
    mismatch_finds = [
        by_location[loc][0] for loc in mismatch_locations
        if loc in by_location and by_location[loc]
    ]
    if mismatch_finds:
        lines.append("## Paraphrase vs equation mismatches")
        lines.append("")
        lines.append("_These flags indicate that the AI's own summary of a claim may not "
                     "match the equation actually printed in the PDF. The referee should "
                     "check whether the discrepancy also exists in the paper text._")
        lines.append("")
        for f in mismatch_finds:
            lines.append(f"- **{f.location}** — [{f.source}] verdict: `{f.verdict}`")
            lines.append(f"  {f.evidence[:1200]}")
            # any *other* findings at the same location — dimensional cross-checks etc.
            others = [g for g in by_location.get(f.location, []) if g is not f]
            for o in others:
                lines.append(f"  · cross-check [{o.source}]: severity=`{o.severity}`, "
                             f"verdict=`{o.verdict}` — {o.evidence[:250]}")
            lines.append("")

    # ---- critical / warning blocks (excluding mismatch locations) ----
    def _primary_and_cross(loc_finds: list[AggregatedFinding]):
        primary = loc_finds[0]
        others = loc_finds[1:]
        return primary, others

    critical_blocks: list[tuple[AggregatedFinding, list[AggregatedFinding]]] = []
    warning_blocks:  list[tuple[AggregatedFinding, list[AggregatedFinding]]] = []
    info_blocks:     list[tuple[AggregatedFinding, list[AggregatedFinding]]] = []

    for loc, finds in by_location.items():
        if loc in mismatch_locations:
            continue
        primary, others = _primary_and_cross(finds)
        if primary.severity == "critical":
            critical_blocks.append((primary, others))
        elif primary.severity == "warning":
            warning_blocks.append((primary, others))
        else:
            info_blocks.append((primary, others))

    def _emit_block(primary: AggregatedFinding, others: list[AggregatedFinding]):
        lines.append(f"- **{primary.location}** — [{primary.source}] `{primary.verdict}`")
        lines.append(f"  {primary.evidence[:1200]}")
        if primary.suggested_fix:
            lines.append(f"  _Suggested fix:_ {primary.suggested_fix}")
        for o in others:
            lines.append(f"  · cross-check [{o.source}]: `{o.verdict}` — {o.evidence[:250]}")
        lines.append("")

    if critical_blocks:
        lines.append("## Critical issues")
        lines.append("")
        for p, o in critical_blocks:
            _emit_block(p, o)

    if warning_blocks:
        lines.append("## Points requiring author clarification")
        lines.append("")
        for p, o in warning_blocks:
            _emit_block(p, o)

    # Only show info block if there are no criticals / warnings / mismatches
    if info_blocks and not (critical_blocks or warning_blocks or mismatch_finds):
        lines.append("## Notes")
        lines.append("")
        lines.append("No dimensional errors or symbolic contradictions were detected. "
                     "Selected passed checks:")
        lines.append("")
        for p, _ in info_blocks[:5]:
            lines.append(f"- **{p.location}** — [{p.source}] `{p.verdict}`")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Generated by PeerLens Math Referee (prototype). Every finding is an "
                 "AI suggestion; a human editor is responsible for the final decision._")
    return "\n".join(lines)


def save_markdown(report: MathReport) -> Path:
    out_dir = get_path("processed_data") / "referee_sections"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}_math.md"
    out_path.write_text(render_markdown(report), encoding="utf-8")
    return out_path


def aggregate_and_render(doi_label: str) -> tuple[Path, Path]:
    """One-shot: aggregate and produce both JSON + markdown."""
    report = aggregate(doi_label)
    json_path = save_math_report(report)
    md_path = save_markdown(report)
    return json_path, md_path
