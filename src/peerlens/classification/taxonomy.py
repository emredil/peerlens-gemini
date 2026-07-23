"""Controlled vocabulary for PeerLens classifier.

Aligned with SciPost Physics `specialties` and `approaches` fields.
Each SciPost specialty maps to a snake_case internal code so we can
extend/rename later without breaking downstream code.
"""
from __future__ import annotations

from typing import Literal


SUBDOMAIN_MAP: dict[str, str] = {
    "Condensed Matter Physics - Theory":                  "cmp_theory",
    "Condensed Matter Physics - Experiment":              "cmp_experiment",
    "Condensed Matter Physics - Computational":           "cmp_computational",
    "High-Energy Physics - Theory":                       "hep_theory",
    "High-Energy Physics - Phenomenology":                "hep_phenomenology",
    "High-Energy Physics - Experiment":                   "hep_experiment",
    "Quantum Physics":                                    "quantum_physics",
    "Mathematical Physics":                               "math_physics",
    "Statistical and Soft Matter Physics":                "statistical_soft",
    "Gravitation, Cosmology and Astroparticle Physics":   "gravitation_cosmo",
    "Atomic, Molecular and Optical Physics - Theory":     "amo_theory",
    "Atomic, Molecular and Optical Physics - Experiment": "amo_experiment",
    "Nuclear Physics - Theory":                           "nuclear_theory",
    "Nuclear Physics - Experiment":                       "nuclear_experiment",
    "Fluid Dynamics":                                     "fluid_dynamics",
}

SUBDOMAIN_CODES: list[str] = sorted(set(SUBDOMAIN_MAP.values()))
SUBDOMAIN_DESCRIPTIONS: dict[str, str] = {v: k for k, v in SUBDOMAIN_MAP.items()}

PaperType = Literal["theoretical", "computational", "experimental", "phenomenological"]
PAPER_TYPES: list[str] = ["theoretical", "computational", "experimental", "phenomenological"]


def normalize_specialties(specialties: list[str]) -> list[str]:
    """Convert SciPost specialty labels to internal codes."""
    return sorted({SUBDOMAIN_MAP[s] for s in specialties if s in SUBDOMAIN_MAP})


def normalize_approaches(approaches_raw: str) -> list[str]:
    """SciPost approaches come as a stringified Python list; parse safely."""
    import ast
    try:
        vals = ast.literal_eval(approaches_raw)
        if isinstance(vals, list):
            return sorted({v for v in vals if v in PAPER_TYPES})
    except (ValueError, SyntaxError):
        pass
    return []
