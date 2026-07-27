"""Controlled vocabulary for PeerLens classifier.

Aligned with SciPost Physics `specialties` and `approaches` fields.
Each SciPost specialty maps to a snake_case internal code so we can
extend/rename later without breaking downstream code.
"""
from __future__ import annotations

from typing import Literal


SUBDOMAIN_MAP: dict[str, str] = {
    # ---- Physics ----
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

    # ---- Mathematics ----
    "Mathematics - Algebra":                              "math_algebra",
    "Mathematics - Algebraic Geometry":                   "math_algebraic_geometry",
    "Mathematics - Analysis":                             "math_analysis",
    "Mathematics - Topology and Geometry":                "math_topology_geometry",
    "Mathematics - Combinatorics":                        "math_combinatorics",
    "Mathematics - Number Theory":                        "math_number_theory",
    "Mathematics - Probability and Stochastic Processes": "math_probability",
    "Mathematics - Logic and Foundations":                "math_logic",

    # ---- Chemistry ----
    "Chemistry - Organic":                                "chem_organic",
    "Chemistry - Inorganic":                              "chem_inorganic",
    "Chemistry - Physical":                               "chem_physical",
    "Chemistry - Computational":                          "chem_computational",
    "Chemistry - Analytical":                             "chem_analytical",
    "Chemistry - Biochemistry and Medicinal":             "chem_biochem_medicinal",

    # ---- Biology ----
    "Biology - Molecular and Genetics":                   "bio_molecular",
    "Biology - Cell and Developmental":                   "bio_cell_dev",
    "Biology - Ecology and Evolution":                    "bio_ecology_evolution",
    "Biology - Neuroscience":                             "bio_neuroscience",
    "Biology - Bioinformatics and Systems":               "bio_bioinformatics_systems",
    "Biology - Biomedicine":                              "bio_biomedicine",
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

# ============================================================
# Field derivation from subdomain code
# ============================================================

Field = Literal["physics", "math", "chemistry", "biology"]
FIELDS: list[str] = ["physics", "math", "chemistry", "biology"]


def field_from_subdomain(subdomain_code: str) -> str:
    """Map a subdomain code to its top-level field.

    Any code prefixed with math_/chem_/bio_ is routed accordingly.
    Everything else (including math_physics, which is physics with
    heavy math formalism) is physics.
    """
    if subdomain_code.startswith("math_") and subdomain_code != "math_physics":
        return "math"
    if subdomain_code.startswith("chem_"):
        return "chemistry"
    if subdomain_code.startswith("bio_"):
        return "biology"
    return "physics"


def field_from_subdomains(subdomains: list) -> str:
    """When a paper has multiple subdomains, pick the field of the first
    subdomain (SciPost editors put primary specialty first)."""
    if not subdomains:
        return "physics"
    return field_from_subdomain(subdomains[0])
