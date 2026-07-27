"""Field-aware rendering of extracted claims for the referee markdown.

Called by referee_aggregator.render_markdown. Kept in a separate module so
the multi-line function bodies do not have to survive heredoc patching in
the aggregator file.
"""
from __future__ import annotations


def render_claims_section(claims_json: dict, field_top: str) -> list:
    """Return markdown lines for the extracted-claims section.

    Layout depends on field: physics, chemistry, math, biology each have
    distinct claim vocabularies. Returns an empty list if there is nothing
    to render.
    """
    lines: list = []
    claims = claims_json.get("claims", []) or []
    symbols = claims_json.get("symbols", []) or []

    if not claims and not symbols:
        return lines

    heading_by_field = {
        "physics":   "## Extracted claims - verifiable equations and symmetries",
        "chemistry": "## Extracted claims - synthesis, characterization, performance",
        "math":      "## Extracted formal statements",
        "biology":   "## Extracted testable claims",
    }
    lines.append(heading_by_field.get(field_top, "## Extracted claims"))
    lines.append("")
    lines.append(f"_{len(claims)} claim(s), {len(symbols)} symbol(s) catalogued._")
    lines.append("")

    if symbols:
        lines.append("**Symbols in play:**")
        lines.append("")
        for s in symbols[:12]:
            sym = s.get("symbol", "")
            mean = s.get("meaning", "")
            units = s.get("units", "")
            loc = s.get("first_seen_at", "")
            units_part = f" [{units}]" if units else ""
            loc_part = f" @ {loc}" if loc else ""
            lines.append(f"- `{sym}`{units_part}: {mean}{loc_part}")
        if len(symbols) > 12:
            lines.append(f"- _...and {len(symbols) - 12} more (see JSON report)_")
        lines.append("")

    by_type: dict = {}
    for c in claims:
        by_type.setdefault(c.get("claim_type", "unknown"), []).append(c)

    if field_top == "math":
        _emit_math_claim_groups(lines, by_type)
    elif field_top == "biology":
        _emit_bio_claim_groups(lines, by_type)
    elif field_top == "chemistry":
        _emit_chem_claim_groups(lines, by_type)
    else:
        _emit_physics_claim_groups(lines, by_type)

    return lines


def _emit_math_claim_groups(lines: list, by_type: dict) -> None:
    order = ["theorem", "proposition", "lemma", "corollary",
             "definition", "criterion", "conjecture"]
    label = {
        "theorem":     "Theorems",
        "proposition": "Propositions",
        "lemma":       "Lemmas",
        "corollary":   "Corollaries",
        "definition":  "Definitions",
        "criterion":   "Criteria",
        "conjecture":  "Conjectures",
    }
    for kind in order:
        items = by_type.get(kind, [])
        if not items:
            continue
        lines.append(f"**{label[kind]}** ({len(items)}):")
        lines.append("")
        for c in items:
            loc = c.get("location", "?")
            nat = (c.get("natural_language") or "")[:400]
            hyp = (c.get("hypothesis") or "")[:250]
            con = (c.get("conclusion") or "")[:250]
            proof = c.get("proof_hint") or ""
            lines.append(f"- `{loc}` - {nat}")
            if hyp:
                lines.append(f"  - _Hypothesis:_ {hyp}")
            if con:
                lines.append(f"  - _Conclusion:_ {con}")
            if proof:
                lines.append(f"  - _Proof technique:_ {proof}")
        lines.append("")


def _emit_bio_claim_groups(lines: list, by_type: dict) -> None:
    order = ["hypothesis", "model_equation", "model_parameter",
             "empirical_measurement", "qualitative_prediction",
             "statistical_claim", "data_availability"]
    label = {
        "hypothesis":             "Mechanistic hypotheses",
        "model_equation":         "Model equations",
        "model_parameter":        "Model parameters",
        "empirical_measurement":  "Empirical measurements",
        "qualitative_prediction": "Qualitative predictions",
        "statistical_claim":      "Statistical claims",
        "data_availability":      "Data and code availability",
    }
    for kind in order:
        items = by_type.get(kind, [])
        if not items:
            continue
        lines.append(f"**{label[kind]}** ({len(items)}):")
        lines.append("")
        for c in items:
            loc = c.get("location", "?")
            nat = (c.get("natural_language") or "")[:400]
            lines.append(f"- `{loc}` - {nat}")
            if c.get("measurement_value"):
                val = c["measurement_value"]
                units = c.get("measurement_units", "")
                unc = c.get("measurement_uncertainty", "")
                unc_part = f" +/- {unc}" if unc else ""
                units_part = f" {units}" if units else ""
                lines.append(f"  - _Value:_ {val}{unc_part}{units_part}")
            if c.get("parameters"):
                params = ", ".join(str(p) for p in c["parameters"][:6])
                lines.append(f"  - _Parameters:_ {params}")
        lines.append("")


def _emit_chem_claim_groups(lines: list, by_type: dict) -> None:
    order = ["synthesis_step", "characterization_result", "performance_metric",
             "calibration_equation", "method_parameter", "selectivity_claim",
             "stability_claim", "validation_result", "mechanism_claim"]
    label = {
        "synthesis_step":          "Synthesis steps",
        "characterization_result": "Characterization results",
        "performance_metric":      "Performance metrics",
        "calibration_equation":    "Calibration equations",
        "method_parameter":        "Method parameters",
        "selectivity_claim":       "Selectivity",
        "stability_claim":         "Stability",
        "validation_result":       "Validation",
        "mechanism_claim":         "Proposed mechanisms",
    }
    for kind in order:
        items = by_type.get(kind, [])
        if not items:
            continue
        lines.append(f"**{label[kind]}** ({len(items)}):")
        lines.append("")
        for c in items:
            loc = c.get("location", "?")
            nat = (c.get("natural_language") or "")[:400]
            lines.append(f"- `{loc}` - {nat}")
            if c.get("measurement_value"):
                val = c["measurement_value"]
                units = c.get("measurement_units", "")
                units_part = f" {units}" if units else ""
                lines.append(f"  - _Value:_ {val}{units_part}")
            if c.get("parameters"):
                params = ", ".join(str(p) for p in c["parameters"][:6])
                lines.append(f"  - _Parameters:_ {params}")
        lines.append("")


def _emit_physics_claim_groups(lines: list, by_type: dict) -> None:
    order = ["dimensional", "algebraic_identity", "limit", "reduction",
             "symmetry", "conservation"]
    label = {
        "dimensional":         "Dimensional consistency",
        "algebraic_identity":  "Algebraic identities",
        "limit":               "Limit / asymptotic behavior",
        "reduction":           "Reductions to known cases",
        "symmetry":            "Symmetry statements",
        "conservation":        "Conservation laws",
    }
    for kind in order:
        items = by_type.get(kind, [])
        if not items:
            continue
        lines.append(f"**{label[kind]}** ({len(items)}):")
        lines.append("")
        for c in items:
            loc = c.get("location", "?")
            nat = (c.get("natural_language") or "")[:400]
            lines.append(f"- `{loc}` - {nat}")
        lines.append("")
