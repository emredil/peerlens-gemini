"""Triage scoring engine.

Converts a paper's multi-agent findings into a single 0-100 quality score
and one of three editorial verdicts:
  - priority   : score >= 70, editor should look now
  - standard   : score 40-69, route to human referees
  - reject     : score < 40, likely garbage, safe to auto-reject subject
                 to editor confirmation

Score construction
------------------
Base: 100. Every finding contributes a signed weight based on its
category (source), kind, and severity. Positive contributions are
possible for genuinely good signals (verified derivations, thorough
literature coverage, originality). Score is clipped to [0, 100].
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

from peerlens.settings import get_path


Verdict = Literal["priority", "standard", "reject", "insufficient_data"]


# ---------------------------------------------------------------------------
# Weight table
# ---------------------------------------------------------------------------
#
# Key format: (source_or_kind, severity). Special "*" matches anything.
# Positive weights reward good signals, negative weights penalize concerns.
# All weights sum on top of a base of 100.

# Severity-driven weights (no positive rewards).
# Score starts at 100 and only decreases. Cap [0, 100].
#
# Design principle: a paper is presumed 'priority' until agents produce
# evidence otherwise. Well-behaved papers stay at 100. Papers with real
# issues accumulate penalties.
#
# Rewarding 'good' signals like sympy_verified or novelty originality
# was removed — those are expected behavior, not bonus territory.

WEIGHTS: dict[tuple[str, str, str], int] = {
    # source, kind, severity  ->  points

    # --- Math ---
    # Genuine sympy refutation is the strongest single signal we have.
    ("math", "sympy_refuted_genuine", "*"):          -25,
    ("math", "dimensional_inconsistent", "critical"): -25,
    ("math", "dimensional_inconsistent", "warning"):  -8,
    ("math", "dimensional_convention_dependent", "warning"): -3,
    ("math", "sympy_verified", "*"):                  0,
    ("math", "sympy_unable", "*"):                    0,
    ("math", "paraphrase_mismatch", "*"):             0,   # system artifact, not paper

    # --- Novelty ---
    ("novelty", "overreach", "critical"):    -25,
    ("novelty", "overreach", "warning"):     -8,
    ("novelty", "missed_citation", "critical"): -25,
    ("novelty", "missed_citation", "warning"):  -8,
    ("novelty", "staleness", "*"):             -3,
    ("novelty", "originality", "*"):           0,           # expected, not rewarded

    # --- Structure ---
    ("structure", "derivation_gap", "critical"):     -25,
    ("structure", "derivation_gap", "warning"):       -8,
    ("structure", "orphan_reference", "*"):           -3,
    ("structure", "handwavy_shortcut", "warning"):    -8,
    ("structure", "handwavy_shortcut", "info"):       0,

    # --- Domain ---
    ("domain", "domain_concern", "critical"):     -25,
    ("domain", "domain_concern", "warning"):       -8,
    ("domain", "assumption_check", "*"):           -3,
    ("domain", "convention_check", "*"):           -3,

    # --- Reproducibility (gate-based, only warning/critical exist) ---
    ("reproducibility", "missing_code", "critical"):     -25,
    ("reproducibility", "missing_code", "warning"):      -8,
    ("reproducibility", "missing_data", "critical"):     -25,
    ("reproducibility", "missing_data", "warning"):      -8,
    ("reproducibility", "missing_params", "warning"):    -8,
    ("reproducibility", "missing_algorithm", "warning"): -8,
}


# Thresholds — conservative first cut.
# 6 published SciPost papers landed at scores 0, 42, 59, 64, 86, 92 with the
# severity-driven weights. Setting priority_min=80 requires clear positive
# signal (few or no critical concerns) and gives 'standard' most middle
# papers. reject_max=39 keeps the reject bar high — only papers with clear
# multiple critical findings drop below.
THRESHOLDS = {
    "priority_min": 80,
    "reject_max":   39,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoreContribution:
    source: str
    kind: str
    severity: str
    location: str
    weight: int
    evidence_short: str


@dataclass
class TriageScore:
    doi_label: str
    score: int = 100
    verdict: Verdict = "insufficient_data"
    n_agent_layers_present: int = 0
    n_positive_contributions: int = 0
    n_negative_contributions: int = 0
    top_negative: list[ScoreContribution] = field(default_factory=list)
    top_positive: list[ScoreContribution] = field(default_factory=list)
    all_contributions: list[ScoreContribution] = field(default_factory=list)
    breakdown_by_source: dict[str, int] = field(default_factory=dict)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _weight_lookup(source: str, kind: str, severity: str) -> int:
    """Try exact (source, kind, sev), then (source, kind, *)."""
    exact = WEIGHTS.get((source, kind, severity))
    if exact is not None:
        return exact
    wild = WEIGHTS.get((source, kind, "*"))
    if wild is not None:
        return wild
    return 0


# Extract "kind"-like tag from math findings (they use `verdict` not `kind`)
def _math_kind_from_finding(f: dict) -> str:
    src = f.get("source", "")            # "dimensional_text" | "dimensional_vision" | "sympy"
    verdict = f.get("verdict", "")       # "consistent" | "inconsistent" | "verified" | "refuted" | ...
    paraphrase_mm = bool(f.get("paraphrase_mismatch", False))

    if src.startswith("dimensional"):
        if verdict == "inconsistent":
            return "dimensional_inconsistent"
        if verdict == "convention_dependent":
            return "dimensional_convention_dependent"
        return "dimensional_other"
    if src == "sympy":
        if verdict == "refuted" and not paraphrase_mm:
            return "sympy_refuted_genuine"
        if paraphrase_mm:
            return "paraphrase_mismatch"
        if verdict == "verified":
            return "sympy_verified"
        if verdict == "unable_to_verify":
            return "sympy_unable"
    return "other"


# ---------------------------------------------------------------------------
# Report loaders
# ---------------------------------------------------------------------------

def _load(doi: str, layer: str) -> Optional[dict]:
    p = get_path("processed_data") / f"{layer}_reports" / f"{doi}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def _short(s: str, n: int = 300) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= n:
        return s
    # Try to break at a sentence boundary before n
    cut = s[:n]
    last_period = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if last_period > n // 2:
        return cut[:last_period + 1]
    return cut + "..."


def _extract_evidence(f: dict, source: str) -> str:
    """Choose the best human-readable evidence field for a finding.

    Sympy findings put the snippet-planning prose in 'evidence' but the
    actual verdict result in 'detail'. For rendering to a human editor
    we prefer 'detail' when it exists, and we look at the finding's
    'human_summary' field first if the agent supplied one.
    """
    hs = f.get("human_summary")
    if hs:
        return hs
    if source == "math":
        # sympy findings: detail is the sandbox result; evidence is planning
        detail = f.get("detail")
        if detail:
            return detail
    return f.get("evidence", "")


def score_paper(doi: str) -> TriageScore:
    result = TriageScore(doi_label=doi)
    contributions: list[ScoreContribution] = []
    breakdown: dict[str, int] = {}

    layers_present = 0

    # --- Math ---
    math = _load(doi, "math")
    if math is not None:
        layers_present += 1
        for f in math.get("findings", []):
            kind = _math_kind_from_finding(f)
            sev = f.get("severity", "info")
            w = _weight_lookup("math", kind, sev)
            if w == 0 and kind != "sympy_unable":
                continue
            c = ScoreContribution(
                source="math",
                kind=kind,
                severity=sev,
                location=f.get("location", ""),
                weight=w,
                evidence_short=_short(_extract_evidence(f, "math")),
            )
            contributions.append(c)
            breakdown["math"] = breakdown.get("math", 0) + w

    # --- Novelty ---
    nov = _load(doi, "novelty")
    if nov is not None:
        layers_present += 1
        for f in nov.get("findings", []):
            kind = f.get("kind", "")
            sev = f.get("severity", "info")
            w = _weight_lookup("novelty", kind, sev)
            if w == 0:
                continue
            c = ScoreContribution(
                source="novelty",
                kind=kind,
                severity=sev,
                location=f.get("location", ""),
                weight=w,
                evidence_short=_short(f.get("evidence", "")),
            )
            contributions.append(c)
            breakdown["novelty"] = breakdown.get("novelty", 0) + w

    # --- Structure ---
    struct = _load(doi, "structure")
    if struct is not None:
        layers_present += 1
        for f in struct.get("findings", []):
            kind = f.get("kind", "")
            sev = f.get("severity", "info")
            w = _weight_lookup("structure", kind, sev)
            if w == 0:
                continue
            c = ScoreContribution(
                source="structure",
                kind=kind,
                severity=sev,
                location=f.get("location", ""),
                weight=w,
                evidence_short=_short(f.get("evidence", "")),
            )
            contributions.append(c)
            breakdown["structure"] = breakdown.get("structure", 0) + w

    # --- Domain ---
    dom = _load(doi, "domain")
    if dom is not None:
        layers_present += 1
        for f in dom.get("findings", []):
            kind = f.get("kind", "")
            sev = f.get("severity", "info")
            w = _weight_lookup("domain", kind, sev)
            if w == 0:
                continue
            c = ScoreContribution(
                source="domain",
                kind=kind,
                severity=sev,
                location=f.get("location", ""),
                weight=w,
                evidence_short=_short(f.get("evidence", "")),
            )
            contributions.append(c)
            breakdown["domain"] = breakdown.get("domain", 0) + w

    # --- Reproducibility ---
    repro = _load(doi, "reproducibility")
    if repro is not None:
        layers_present += 1
        for f in repro.get("findings", []):
            kind = f.get("kind", "")
            sev = f.get("severity", "warning")
            w = _weight_lookup("reproducibility", kind, sev)
            if w == 0:
                continue
            c = ScoreContribution(
                source="reproducibility",
                kind=kind,
                severity=sev,
                location=f.get("location", ""),
                weight=w,
                evidence_short=_short(f.get("evidence", "")),
            )
            contributions.append(c)
            breakdown["reproducibility"] = breakdown.get("reproducibility", 0) + w

    result.n_agent_layers_present = layers_present

    if layers_present == 0:
        result.rationale = "No agent reports found for this paper."
        return result

    total_delta = sum(c.weight for c in contributions)
    raw_score = 100 + total_delta
    result.score = max(0, min(100, raw_score))

    result.all_contributions = contributions
    result.n_positive_contributions = sum(1 for c in contributions if c.weight > 0)
    result.n_negative_contributions = sum(1 for c in contributions if c.weight < 0)

    negatives = sorted([c for c in contributions if c.weight < 0], key=lambda c: c.weight)[:5]
    positives = sorted([c for c in contributions if c.weight > 0], key=lambda c: -c.weight)[:5]
    result.top_negative = negatives
    result.top_positive = positives
    result.breakdown_by_source = breakdown

    # --- Verdict ---
    if result.score >= THRESHOLDS["priority_min"]:
        result.verdict = "priority"
    elif result.score <= THRESHOLDS["reject_max"]:
        result.verdict = "reject"
    else:
        result.verdict = "standard"

    return result


def save_triage_score(result: TriageScore) -> Path:
    out_dir = get_path("processed_data") / "triage_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    return out_path
