"""Claim Extractor — field-conditional.

Physics + chemistry: original schema (dimensional, limit, reduction,
algebraic_identity, symmetry, conservation). Downstream sympy and
dimensional_checker are wired for this vocabulary.

Mathematics: theorem / lemma / proposition / definition / corollary /
criterion, each with hypothesis + conclusion + proof structure. Not
consumed by sympy (formal proofs are outside its scope) but drives
domain expert review and the markdown report.

Biology: hypothesis / model_equation / model_parameter /
empirical_measurement / qualitative_prediction / data_availability.
Same disposition — not sympy-verified, but structured for the
domain expert and report.

The four schemas share the outer envelope (symbols + claims) so
downstream code that only reads doi_label + n_claims_total continues
to work unchanged."""
from __future__ import annotations

from typing import Any

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


@dataclass
class MathClaim:
    claim_type: str
    location: str
    referenced_equations: list[str]
    natural_language: str
    check_hint: str
    verification_method: str
    # Field-specific optional fields — populated by non-physics extractors.
    hypothesis: str = ""
    conclusion: str = ""
    proof_hint: str = ""
    parameters: list = None
    measurement_value: str = ""
    measurement_units: str = ""
    measurement_uncertainty: str = ""

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []


@dataclass
class SymbolDefinition:
    symbol: str
    meaning: str
    units: str
    first_seen_at: str


@dataclass
class ExtractedClaims:
    doi_label: str
    claims: list[MathClaim] = field(default_factory=list)
    symbols: list[SymbolDefinition] = field(default_factory=list)
    n_claims_total: int = 0
    n_dimensional: int = 0
    n_limit: int = 0
    n_reduction: int = 0
    n_algebraic: int = 0
    n_symmetry: int = 0
    n_conservation: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "symbols": {
            "type": "array",
            "minItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "meaning": {"type": "string"},
                    "units": {"type": "string"},
                    "first_seen_at": {"type": "string"},
                },
                "required": ["symbol", "meaning", "units"],
            },
        },
        "claims": {
            "type": "array",
            "minItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "claim_type": {
                        "type": "string",
                        "enum": ["dimensional", "limit", "reduction", "algebraic_identity", "symmetry", "conservation"],
                    },
                    "location": {"type": "string"},
                    "referenced_equations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "natural_language": {"type": "string"},
                    "check_hint": {"type": "string"},
                    "verification_method": {
                        "type": "string",
                        "enum": ["sympy", "vision", "literature", "manual"],
                    },
                },
                "required": ["claim_type", "location", "natural_language", "check_hint", "verification_method"],
            },
        },
    },
    "required": ["symbols", "claims"],
}


SYSTEM_PROMPT = """You are a mathematical claim extractor for physics manuscripts.

Your job is to read a paper and surface every mathematical statement that a
rigorous human referee would want to verify. You do NOT verify claims; you
list them so a downstream agent can check them.

CATEGORIES (be precise about which one applies):

1. dimensional: Both sides of an equation must share physical dimensions.
2. limit: A specific behaviour claimed as a parameter approaches a value.
3. reduction: The result specialises to a known established result in some regime.
4. algebraic_identity: An equation asserted to hold (commutator vanishes, two
   operators equal, an operator squares to a scalar, etc.). Most common in
   algebra-heavy papers. Prefer this over 'symmetry' for concrete equations.
5. symmetry: Invariance under a transformation (Lorentz, gauge, translation).
   Do NOT tag every commutation relation as symmetry.
6. conservation: A quantity conserved by dynamics. Example: [H, Q] = 0.

VERIFICATION METHOD:
- sympy: symbolic algebra can decide it — commutators, substitutions, limits.
- vision: must read PDF page image to see the exact equation.
- literature: verification requires cross-checking a cited paper.
- manual: needs human judgment.

SYMBOLS: You MUST catalogue at least 5 key symbols involved in the claims.
Fill in the `symbols` array FIRST, before the `claims` array. This is
required because the tool call is bounded in size and if you write symbols
last, they may get truncated. The order must be: symbols then claims.

PRINCIPLES:
- Be exhaustive for the categories, not exhaustive in general.
- Use the author's language where possible.
- If a claim cannot be verified in principle, omit it.
- Prefer algebraic_identity over symmetry for concrete commutation statements.

Use LaTeX notation for any mathematical symbols and equations in your output. Write `$\\varepsilon$` not `ε`, `$\\mu$` not `µ`, `$\\pi$` not `π`. Every mathematical symbol must be wrapped in `$...$`. Never use Unicode math characters."""


MATH_SCHEMA = {
    "type": "object",
    "properties": {
        "symbols": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "meaning": {"type": "string"},
                    "units": {"type": "string"},
                    "first_seen_at": {"type": "string"},
                },
                "required": ["symbol", "meaning", "first_seen_at"],
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_type": {
                        "type": "string",
                        "enum": ["theorem", "lemma", "proposition", "corollary",
                                 "definition", "criterion", "conjecture"],
                    },
                    "location": {"type": "string"},
                    "natural_language": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "conclusion": {"type": "string"},
                    "proof_hint": {"type": "string"},
                    "referenced_equations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "check_hint": {"type": "string"},
                    "verification_method": {
                        "type": "string",
                        "enum": ["formal_proof", "counterexample_search",
                                 "reference_check", "unverifiable_by_tool"],
                    },
                },
                "required": ["claim_type", "location", "natural_language",
                             "hypothesis", "conclusion", "verification_method"],
            },
        },
    },
    "required": ["symbols", "claims"],
}


MATH_SYSTEM_PROMPT = """You are a claim extractor for pure-mathematics manuscripts.
Your job is to catalogue the paper's formal statements so that a domain expert
can assess them efficiently.

A formal statement is anything the author labeled as one of:
theorem, lemma, proposition, corollary, definition, criterion, conjecture.
Extract each such statement as one claim.

FIELDS TO POPULATE FOR EACH CLAIM:

- claim_type: which of the labels above the author used.
- location: the exact label the paper gave it (e.g. "Theorem 3.2",
  "Lemma 4.4", "Definition 2.8"). Use whatever numbering the paper uses.
- natural_language: a one-sentence summary in plain English.
- hypothesis: the "if" / "let" / "suppose" side of the statement —
  the assumptions, the setup, the objects the statement talks about.
  For a definition, put the term being defined here (the definiendum).
- conclusion: the "then" side — what the statement asserts under those
  hypotheses. For a definition, put the defining condition here (the
  definiens).
- proof_hint: at most one short phrase describing the proof technique if
  the paper hints at it. Typical hints across pure math include
  induction, universal covers, spectral sequences, moment maps,
  compactness arguments, characteristic classes, explicit computation,
  cited black-box theorem, or a specific named lemma. Leave empty if
  the paper does not indicate a technique for this statement.
- referenced_equations: labels of equations invoked in the hypothesis or
  conclusion, if the paper numbers them.
- verification_method: one of
    formal_proof: statement is a candidate for Lean/Coq or manual
      referee review of the given proof.
    counterexample_search: statement is universally quantified ("for
      all X, P(X)") so a single counterexample would refute it.
    reference_check: the claim rests on a cited prior result that a
      referee would want to spot-check.
    unverifiable_by_tool: philosophical, foundational, or highly
      abstract claim that no tool this pipeline offers can address.

DEFINITIONS COUNT AS CLAIMS. Definitions constrain everything that
follows; treat them as first-class formal statements.

SYMBOLS: catalogue at least 5 key notational objects the paper
introduces or reuses (spaces, maps, group elements, invariants, moduli,
functors, ...). For each: meaning and location where introduced. The
'units' field can be empty — pure math is typically dimensionless.

STRICT RULES:
- Do NOT fabricate. If the paper is short and has few labeled
  statements, return few claims rather than invented ones.
- Do NOT extract prose observations, motivation paragraphs, or
  informal remarks as claims. Only formal labeled statements.
- The JSON output order MUST be: symbols first, then claims.

Aim for 5-15 claims for a typical paper; extended surveys may go
higher, short notes lower.

Use LaTeX notation for any mathematical symbols and equations in your output. Write `$\\varepsilon$` not `ε`, `$\\mu$` not `µ`, `$\\pi$` not `π`. Every mathematical symbol must be wrapped in `$...$`. Never use Unicode math characters."""


BIO_SCHEMA = {
    "type": "object",
    "properties": {
        "symbols": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "meaning": {"type": "string"},
                    "units": {"type": "string"},
                    "first_seen_at": {"type": "string"},
                },
                "required": ["symbol", "meaning", "first_seen_at"],
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_type": {
                        "type": "string",
                        "enum": ["hypothesis", "model_equation", "model_parameter",
                                 "empirical_measurement", "qualitative_prediction",
                                 "data_availability", "statistical_claim"],
                    },
                    "location": {"type": "string"},
                    "natural_language": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "conclusion": {"type": "string"},
                    "parameters": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "measurement_value": {"type": "string"},
                    "measurement_units": {"type": "string"},
                    "measurement_uncertainty": {"type": "string"},
                    "referenced_equations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "check_hint": {"type": "string"},
                    "verification_method": {
                        "type": "string",
                        "enum": ["reproduce_experiment", "reanalyze_data",
                                 "statistical_recheck", "reference_check",
                                 "model_resimulation", "unverifiable_by_tool"],
                    },
                },
                "required": ["claim_type", "location", "natural_language",
                             "verification_method"],
            },
        },
    },
    "required": ["symbols", "claims"],
}

BIO_SYSTEM_PROMPT = """You are a claim extractor for biology and biophysics
manuscripts covering any of: molecular biology, cell/developmental biology,
ecology and evolution, neuroscience, bioinformatics/systems biology,
biomedicine. Your job is to catalogue the paper's testable claims so that a
domain expert can assess them efficiently.

FIELDS TO POPULATE FOR EACH CLAIM:

- claim_type: one of
    hypothesis: a mechanistic assertion about how a biological system
      or process works. Fill hypothesis (the mechanism / conditions)
      and conclusion (the observable consequence).
    model_equation: a specific mathematical model of a biological
      process (ODE / PDE system, network model, statistical / graphical
      model, dose-response, kinetic law). Give the equation form in
      natural_language and referenced_equations if the paper numbers them.
    model_parameter: a specific parameter value used in simulations or
      fits (rate constants, coupling strengths, sample sizes, regime
      values, sweeping ranges). Put the parameter name(s) in the
      parameters array and the setting / value in natural_language.
    empirical_measurement: a quantitative result from data or
      simulation. Fill measurement_value, measurement_units,
      measurement_uncertainty. Put sample size and experimental
      conditions in natural_language.
    qualitative_prediction: a behavior the model or theory predicts
      as a parameter or condition changes ("as X increases, the system
      transitions from A to B to C"). Fill hypothesis (regime /
      condition) and conclusion (behavior).
    data_availability: a concrete deposit of code, data, sequences,
      protocols, or media. Put URLs, DOIs, or accession numbers in
      natural_language.
    statistical_claim: a specific statistical assertion (test used,
      p-value, R-squared, confidence interval, replicate count, effect
      size). Include the exact numeric value and the test in
      natural_language.

- location: where in the paper the claim is made (section number and
  heading, figure caption number, table number). Be specific.
- verification_method: one of
    reproduce_experiment: requires wet-lab or field replication.
    reanalyze_data: raw data is available; recompute from source.
    statistical_recheck: rerun the reported statistical test.
    reference_check: claim rests on cited prior result.
    model_resimulation: rerun the computational model to check the
      reported behavior. Prefer this for computational-biology papers
      whose central object is a simulation rather than a wet experiment.
    unverifiable_by_tool: narrative, philosophical, or highly abstract
      claim not addressable by this pipeline.

SYMBOLS: catalogue at least 5 domain-specific quantities or entities the
paper uses (receptors, ions, cell types, gene/protein names,
concentrations, coupling constants, model variables, statistical
quantities). Give meaning; give units when applicable (dimensionless
is fine for ratios, phases, or index variables); give location where
first introduced.

STRICT RULES:
- Do NOT fabricate. If a category is absent (e.g. no statistical claim
  because the paper is purely mechanistic), simply return zero items
  in that category.
- Do NOT extract background paragraphs or motivation prose as claims.
  Only concrete assertions the paper makes about its own system,
  model, or data.
- If the paper is COMPUTATIONAL biology (models, simulations, no
  wet-lab component), prefer verification_method="model_resimulation"
  over "reproduce_experiment" for model-based claims.
- The JSON output order MUST be: symbols first, then claims.

Aim for 6-15 claims for a full paper; short reports may have fewer,
comprehensive studies may have more.

Use LaTeX notation for any mathematical symbols and equations in your output. Write `$\\varepsilon$` not `ε`, `$\\mu$` not `µ`, `$\\pi$` not `π`. Every mathematical symbol must be wrapped in `$...$`. Never use Unicode math characters."""

CHEM_SCHEMA = {
    "type": "object",
    "properties": {
        "symbols": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "meaning": {"type": "string"},
                    "units": {"type": "string"},
                    "first_seen_at": {"type": "string"},
                },
                "required": ["symbol", "meaning", "first_seen_at"],
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_type": {
                        "type": "string",
                        "enum": ["synthesis_step", "characterization_result",
                                 "performance_metric", "calibration_equation",
                                 "method_parameter", "selectivity_claim",
                                 "stability_claim", "validation_result",
                                 "mechanism_claim"],
                    },
                    "location": {"type": "string"},
                    "natural_language": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "conclusion": {"type": "string"},
                    "parameters": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "measurement_value": {"type": "string"},
                    "measurement_units": {"type": "string"},
                    "measurement_uncertainty": {"type": "string"},
                    "referenced_equations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "check_hint": {"type": "string"},
                    "verification_method": {
                        "type": "string",
                        "enum": ["reproduce_synthesis", "recheck_characterization",
                                 "algebraic_identity", "dimensional",
                                 "statistical_recheck", "reference_check",
                                 "unverifiable_by_tool"],
                    },
                },
                "required": ["claim_type", "location", "natural_language",
                             "verification_method"],
            },
        },
    },
    "required": ["symbols", "claims"],
}

CHEM_SYSTEM_PROMPT = """You are a claim extractor for chemistry manuscripts
covering any of: organic, inorganic / organometallic, physical /
spectroscopic, computational (DFT / MD), analytical / sensor, biochemistry /
medicinal chemistry. Your job is to catalogue the paper's testable claims
so that a domain expert can assess them efficiently.

FIELDS TO POPULATE FOR EACH CLAIM:

- claim_type: one of
    synthesis_step: a reproducible procedure step. Put reagents,
      solvents, conditions (temperature, time, atmosphere), and any
      workup / purification in natural_language. Put quantitative
      settings in the parameters array.
    characterization_result: an analytical data point identifying or
      quantifying the material or product. Typical techniques include
      XRD, XPS, EXAFS, EELS, NMR, IR, Raman, UV-vis, ICP-MS, TGA,
      DSC, SEM/TEM, HAADF-STEM, EDS, cyclic voltammetry, EIS, DLS,
      BET. Name the technique in natural_language; fill
      measurement_value and measurement_units.
    performance_metric: a headline application-level number tied to
      the paper's central result (limit of detection, yield, ee/dr,
      turnover number, quantum yield, conversion, selectivity ratio,
      catalytic activity, capacity, cycle life, ...). Name the
      analyte or reaction in natural_language if relevant.
    calibration_equation: a regression / calibration formula the
      paper reports (linear fit, Langmuir isotherm, Randles-Sevcik,
      Michaelis-Menten, Arrhenius, van't Hoff, Eyring, Hill equation,
      Beer-Lambert). Put the equation in natural_language and, when
      numbered, in referenced_equations. ALWAYS set
      verification_method="algebraic_identity" for this claim_type.
    method_parameter: a measurement or instrument setting (electrolyte
      composition and pH, scan rate, injection volume, gradient
      program, temperature range, integration time, detection
      criterion). Put settings in parameters and role in natural_language.
    selectivity_claim: a statement about signal or reaction
      specificity in the presence of interferents or competing
      substrates.
    stability_claim: a time-course or cycle-endurance statement.
      Fill measurement_value and measurement_uncertainty (RSD) when
      the paper gives them.
    validation_result: a cross-validation number against a reference
      method or a standard. Name the reference method in
      natural_language.
    mechanism_claim: a proposed catalytic, detection, or reaction
      mechanism. These typically need experimental or computational
      corroboration.

- location: section or figure/table where the claim appears.
- verification_method: one of
    reproduce_synthesis: someone must remake the material.
    recheck_characterization: reanalyze the raw spectral / imaging
      data.
    algebraic_identity: a formal equation whose form can be checked
      with symbolic tools. USE THIS for calibration_equation.
    dimensional: units must be consistent (relevant when a metric is
      tied to physical constants).
    statistical_recheck: rerun statistical analysis on given data.
    reference_check: claim rests on cited prior result.
    unverifiable_by_tool: mechanism proposals typically fall here.

SYMBOLS: catalogue at least 5 domain-specific entities the paper
uses (materials, reagents, catalysts, catalytic sites, ligands, key
intermediates, instrumental quantities). Give meaning, units when
applicable, and location where first introduced.

STRICT RULES:
- Do NOT fabricate. If the paper lacks a category (e.g. no calibration
  equation), simply return zero items in that category.
- Do NOT extract prose background or literature summary as claims.
  Only concrete assertions the paper makes about its own materials,
  measurements, or performance.
- For claim_type="calibration_equation" ALWAYS set
  verification_method="algebraic_identity" so downstream symbolic
  tools can attempt a consistency check.
- The JSON output order MUST be: symbols first, then claims.

Aim for 8-20 claims for a full paper covering synthesis, characterization,
and performance; smaller studies may have fewer.

Use LaTeX notation for any mathematical symbols and equations in your output. Write `$\\varepsilon$` not `ε`, `$\\mu$` not `µ`, `$\\pi$` not `π`. Every mathematical symbol must be wrapped in `$...$`. Never use Unicode math characters."""


def _format_paper(parsed: dict) -> str:
    sec_body = []
    for s in parsed.get("sections", [])[:20]:
        body = s["text"][:4000]
        sec_body.append(f"### {s['number']} {s['title']}\n{body}")
    section_text = "\n\n".join(sec_body)
    return f"""PAPER: {parsed.get('title', '')}

ABSTRACT:
{parsed.get('abstract', '')}

BODY (sections):
{section_text}

METADATA:
  equation_count: {parsed.get('equation_count', 0)}
  figure_count: {len(parsed.get('figure_captions', []))}
  n_pages: {parsed.get('n_pages', 0)}"""


def extract_claims(parsed_paper: dict, client: GeminiClient | None = None,
                    field: str = "physics", uploaded_file: Any = None) -> ExtractedClaims:
    if client is None:
        client = GeminiClient()
    model = CONFIG["gemini"]["model_classifier"]

    if field == "math":
        active_system = MATH_SYSTEM_PROMPT
        active_schema = MATH_SCHEMA
        active_tool_name = "record_math_paper_claims"
        active_tool_desc = "Record theorems, lemmas, propositions, definitions of a pure-math paper."
        active_user_prefix = "Extract formal statements from this mathematics paper."
    elif field == "biology":
        active_system = BIO_SYSTEM_PROMPT if BIO_SYSTEM_PROMPT else SYSTEM_PROMPT
        active_schema = BIO_SCHEMA if BIO_SCHEMA else CLAIM_SCHEMA
        active_tool_name = "record_bio_claims"
        active_tool_desc = "Record hypotheses, model equations, and measurements of a biology paper."
        active_user_prefix = "Extract structured claims from this biology paper."
    elif field == "chemistry":
        active_system = CHEM_SYSTEM_PROMPT if CHEM_SYSTEM_PROMPT else SYSTEM_PROMPT
        active_schema = CHEM_SCHEMA if CHEM_SCHEMA else CLAIM_SCHEMA
        active_tool_name = "record_chem_claims"
        active_tool_desc = "Record synthesis, characterization, and performance claims of a chemistry paper."
        active_user_prefix = "Extract structured claims from this chemistry paper."
    else:
        active_system = SYSTEM_PROMPT
        active_schema = CLAIM_SCHEMA
        active_tool_name = "record_math_claims"
        active_tool_desc = "Record all verifiable mathematical claims AND catalogue at least 5 key symbols."
        active_user_prefix = "Extract math claims from this paper."

    result = client.structured_call(
        model=model,
        system=active_system,
        user=f"{active_user_prefix}\n\n{_format_paper(parsed_paper)}",
        tool_name=active_tool_name,
        tool_description=active_tool_desc,
        tool_input_schema=active_schema,
        max_output_tokens=8192,
        uploaded_file=uploaded_file,
    )

    payload = result.tool_input
    if field == "physics":
        valid_types = {"dimensional", "limit", "reduction", "algebraic_identity", "symmetry", "conservation"}
        for c in payload.get("claims", []):
            if c.get("claim_type") not in valid_types:
                c["claim_type"] = "algebraic_identity"

    claims = [MathClaim(
        claim_type=c["claim_type"],
        location=c.get("location", ""),
        referenced_equations=c.get("referenced_equations", []),
        natural_language=c["natural_language"],
        check_hint=c.get("check_hint", ""),
        verification_method=c.get("verification_method", "manual"),
        hypothesis=c.get("hypothesis", ""),
        conclusion=c.get("conclusion", ""),
        proof_hint=c.get("proof_hint", ""),
        parameters=c.get("parameters", []) or [],
        measurement_value=c.get("measurement_value", ""),
        measurement_units=c.get("measurement_units", ""),
        measurement_uncertainty=c.get("measurement_uncertainty", ""),
    ) for c in payload.get("claims", [])]

    symbols = [SymbolDefinition(
        symbol=s["symbol"],
        meaning=s["meaning"],
        units=s.get("units", ""),
        first_seen_at=s.get("first_seen_at", ""),
    ) for s in payload.get("symbols", [])]

    return ExtractedClaims(
        doi_label=parsed_paper["doi_label"],
        claims=claims,
        symbols=symbols,
        n_claims_total=len(claims),
        n_dimensional=sum(1 for c in claims if c.claim_type == "dimensional"),
        n_limit=sum(1 for c in claims if c.claim_type == "limit"),
        n_reduction=sum(1 for c in claims if c.claim_type == "reduction"),
        n_algebraic=sum(1 for c in claims if c.claim_type == "algebraic_identity"),
        n_symmetry=sum(1 for c in claims if c.claim_type == "symmetry"),
        n_conservation=sum(1 for c in claims if c.claim_type == "conservation"),
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def save_claims(result: ExtractedClaims, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "claims"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    return out_path
