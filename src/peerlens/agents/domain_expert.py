"""Domain Expert Agent (Gemini port).

Given a paper's subdomain classification, this agent applies a
subdomain-specific referee persona and checklist to surface concerns a
generic reviewer would miss.

The domain persona is selected from a prompt bank indexed by the codes
produced by peerlens.classification.taxonomy. If a paper has multiple
subdomains the agent focuses on the primary one; a small set of universal
checks always applies as well.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


@dataclass
class DomainFinding:
    kind: str          # "domain_concern" | "assumption_check" | "convention_check" | "info"
    severity: str      # "critical" | "warning" | "info"
    location: str
    evidence: str
    suggested_fix: str = ""
    confidence: float = 0.0


@dataclass
class DomainReport:
    doi_label: str
    subdomain_used: str = ""
    persona: str = ""
    n_findings: int = 0
    n_critical: int = 0
    n_warning: int = 0
    n_info: int = 0
    findings: list[DomainFinding] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Prompt bank
# ---------------------------------------------------------------------------

DOMAIN_PROMPTS = {
    "gravitation_cosmo": {
        "persona": "gravitational-wave / cosmology theorist",
        "checklist": [
            "Is the post-Newtonian or post-Minkowskian expansion order stated and consistent throughout?",
            "Are gauge choices made explicit (harmonic, TT, radiation, etc.) and used consistently?",
            "For inflationary work: is the slow-roll approximation regime clearly stated? Are Ω/H, ε, η kept small where claimed?",
            "Is backreaction on the background metric explicitly justified as negligible?",
            "For binary systems: are mass-ratio expansion terms tracked to the stated order? Are self-force or spin-orbit couplings addressed if relevant?",
            "Are UV completions / EFT cutoffs discussed for effective descriptions?",
            "For axion / moduli / inflaton fields: is the field range compared with M_Pl? Are corrections beyond leading order in the ratio addressed?",
            "Does the paper claim exact results where only leading-order should be trusted?",
        ],
    },
    "cmp_theory": {
        "persona": "condensed matter theorist",
        "checklist": [
            "Are boundary conditions (periodic, open, twisted) specified and used consistently?",
            "Is the thermodynamic limit taken explicitly where claimed? Are finite-size corrections addressed?",
            "For spin chains / lattice models: is integrability or exact solvability supported by a known algebraic structure (Yang-Baxter, Bethe ansatz, free fermion)?",
            "Are symmetry claims (U(1), SU(2), translation, time-reversal) matched to concrete generators or transformations?",
            "For strongly-correlated systems: is the perturbative regime (large-N, weak coupling, small filling) stated?",
            "Are gap / gapless assertions supported by spectrum arguments or numerical evidence?",
            "For quantum phase transitions: are the order parameters and critical exponents explicit?",
            "Are conserved quantities constructed explicitly rather than asserted?",
        ],
    },
    "hep_theory": {
        "persona": "high-energy theory referee",
        "checklist": [
            "Are renormalization scheme and scale choices stated?",
            "For string theory / SUSY: is the vacuum / brane configuration identified? Are moduli stabilized or are they left open?",
            "For SCFT / integrable systems: is the symmetry algebra used (superconformal, integrable) matched to a concrete realization?",
            "Are dualities used (S-, T-, Seiberg, holographic) invoked with the correct regimes of validity?",
            "For lattice / non-perturbative claims: is the continuum limit or non-perturbative definition clear?",
            "Are anomaly-matching or 't Hooft anomaly conditions checked where relevant?",
            "For asymptotic safety / UV completion: is the ambiguity in choice of couplings addressed?",
            "Are Wick rotations / analytic continuations justified?",
        ],
    },
    "math_physics": {
        "persona": "mathematical physicist",
        "checklist": [
            "Are theorems stated with hypotheses, and are corner cases addressed?",
            "For integrable systems: is the algebraic structure (Lie algebra, Hopf algebra, Yangian) identified and used consistently?",
            "Are functional-analytic subtleties (domain, self-adjointness, closure) addressed where they matter?",
            "For rigorous claims: is the proof self-contained or are the citations to prior work sufficient?",
            "Are counterexamples to sharpness of claims discussed?",
            "For semigroup / operator methods: are boundedness, positivity, spectral gap addressed?",
            "Is the difference between formal series and convergent expansions explicit?",
            "Are graph-theoretic or combinatorial constructions rigorously defined before use?",
        ],
    },
    "statistical_soft": {
        "persona": "statistical / soft-matter theorist",
        "checklist": [
            "Are non-equilibrium assumptions (detailed balance breaking, driving) explicit?",
            "For hydrodynamics / kinetic theory: is the coarse-graining scale stated? Are Chapman-Enskog or GHD expansions used to the correct order?",
            "Are fluctuation theorems, large deviations, or generating functionals invoked with correct convexity / regularity conditions?",
            "For stochastic processes: is the noise correlator specified (Gaussian, correlated, multiplicative)?",
            "Are conservation laws (mass, momentum, energy) tracked through coarse-graining?",
            "For active-matter / driven systems: is the entropy production accounted for?",
            "Is the crossover from ballistic to diffusive to superdiffusive regimes handled properly?",
            "Are numerical simulations of stochastic dynamics validated against known limits?",
        ],
    },
    "quantum_physics": {
        "persona": "quantum information / measurement theorist",
        "checklist": [
            "Are measurement / decoherence models specified concretely?",
            "For quantum error correction / codes: are the distance, threshold, and code structure explicit?",
            "Are entanglement measures (von Neumann, Renyi, negativity) defined with the correct partial trace?",
            "For quantum simulations: is the mapping to physical qubits / gates concrete?",
            "Are fidelity / distance claims stated with the metric (trace distance, fidelity, diamond norm)?",
            "For open-system dynamics: are Markovian assumptions justified?",
            "For quantum thermodynamics: are work / heat definitions consistent?",
        ],
    },
}


DEFAULT_UNIVERSAL = {
    "persona": "generalist physics referee",
    "checklist": [
        "Are the paper's central claims stated as theorems / propositions or as arguments?",
        "Are the assumptions under which each main result holds stated near the result?",
        "Are limiting or degenerate cases addressed?",
        "Is the paper's contribution stated in one sentence in the introduction?",
    ],
}


# ---------------------------------------------------------------------------
# Prompt build + agent
# ---------------------------------------------------------------------------

DOMAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["domain_concern", "assumption_check", "convention_check", "info"],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "warning", "info"],
                    },
                    "location": {"type": "string"},
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["kind", "severity", "evidence", "confidence"],
            },
        },
    },
    "required": ["findings"],
}


DOMAIN_SYSTEM_TEMPLATE = """You are a senior physics referee acting as a specialist in
{persona}. You have been shown a manuscript's title, abstract, section
headings, and introduction. Apply the specialist checklist below to
identify substantive concerns the generic reviewer would miss.

SUBDOMAIN CHECKLIST:
{checklist}

FINDING KINDS:
- domain_concern: A substantive issue specific to this subdomain
  (e.g., missing gauge specification, unclear PN order, ignored
  backreaction, undefined boundary conditions).
- assumption_check: An assumption made implicitly that should be
  stated explicitly (e.g., slow-roll validity, weak-coupling limit).
- convention_check: A convention that is standard in the field but
  the paper does not confirm (e.g., natural units, metric signature,
  choice of gauge).
- info: A descriptive observation, not requiring author action.

SEVERITY:
- critical: A central claim rests on the unaddressed issue.
- warning: The referee would ask the author to clarify.
- info: A note; no action required.

STRICTNESS:

Only raise concerns for which the paper as summarised gives you concrete
evidence of the omission. If the abstract or introduction addresses the
issue, do not flag it. Prefer 3-6 focused findings over a long list.

If the paper falls outside your subdomain expertise (mismatched subdomain
routing), return zero findings with a single 'info' note saying so."""


def _classification_for(doi: str) -> Optional[dict]:
    path = get_path("processed_data") / "classifications" / f"{doi}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _pick_primary_subdomain(classification: Optional[dict]) -> str:
    """Return the primary subdomain code, or 'universal' if unknown."""
    if not classification:
        return "universal"
    subs = classification.get("subdomains", []) or []
    for s in subs:
        if s in DOMAIN_PROMPTS:
            return s
    return "universal"


def _build_system_prompt(subdomain: str) -> tuple[str, str]:
    """Return (persona, full_system_prompt) for the chosen subdomain."""
    if subdomain in DOMAIN_PROMPTS:
        block = DOMAIN_PROMPTS[subdomain]
    else:
        block = DEFAULT_UNIVERSAL
    checklist = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(block["checklist"]))
    return block["persona"], DOMAIN_SYSTEM_TEMPLATE.format(
        persona=block["persona"], checklist=checklist,
    )


def _format_paper(parsed: dict) -> str:
    intro = ""
    for s in parsed.get("sections", []):
        num = str(s.get("number", ""))
        if num.startswith("1") or "intro" in (s.get("title") or "").lower():
            intro = s.get("text", "")[:3500]
            break

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

INTRODUCTION (excerpt):
{intro}

EQUATIONS: {parsed.get('equation_count', 0)}   PAGES: {parsed.get('n_pages', 0)}"""


def analyze_domain(doi: str, client: Optional[GeminiClient] = None) -> DomainReport:
    if client is None:
        client = GeminiClient()

    report = DomainReport(doi_label=doi)

    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not parsed_path.exists():
        return report
    parsed = json.loads(parsed_path.read_text())

    classification = _classification_for(doi)
    subdomain = _pick_primary_subdomain(classification)
    report.subdomain_used = subdomain

    persona, system_prompt = _build_system_prompt(subdomain)
    report.persona = persona

    user = f"""PAPER CONTEXT:

{_format_paper(parsed)}


Apply your specialist checklist. Return concrete findings only.
If a checklist item is satisfied by what you see, do NOT create an
'info' finding for it — silence is the default."""

    result = client.structured_call(
        model=CONFIG["gemini"]["model_specialist"],
        system=system_prompt,
        user=user,
        tool_name="record_domain_findings",
        tool_description="Record subdomain-specific referee findings for the manuscript.",
        tool_input_schema=DOMAIN_SCHEMA,
        max_output_tokens=3500,
        fallback_models=CONFIG["gemini"].get("fallback_chain", []),
    )

    for f in result.tool_input.get("findings", []):
        report.findings.append(DomainFinding(
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
    report.n_info = sum(1 for f in report.findings if f.severity == "info")
    report.model = result.model
    report.input_tokens = result.input_tokens
    report.output_tokens = result.output_tokens
    return report


def save_report(report: DomainReport, out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "domain_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
