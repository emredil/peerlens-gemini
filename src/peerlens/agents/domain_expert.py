"""Domain Expert Agent — silence-default.

Given a paper's subdomain classification, this agent applies a
subdomain-specific referee persona and checklist to surface concerns a
generic reviewer would miss.

Design (post-variance-diagnostic):
  - Silence is the default. Empty findings list is preferable to filler.
  - No soft quota — the previous "Prefer 3-6 focused findings" line
    drove fabrication and 0% strict cross-run stability on a 6x3 probe.
  - Kind enum narrowed to 2 categories (from 4).
  - Severity enum narrowed to critical/warning (info removed).
  - Location + evidence must be explicit direct quotes.
  - Post-hoc dedup on (location, kind).
  - Post-hoc confidence filter (< 0.5 dropped).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


@dataclass
class DomainFinding:
    kind: str          # "domain_concern"
    severity: str      # "critical" | "warning"
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
    findings: list[DomainFinding] = field(default_factory=list)
    strengths_narrative: str = ""
    weaknesses_narrative: str = ""
    commentary_narrative: str = ""
    verdict: str = ""  # "accept" | "minor revision" | "major revision" | "reject"
    verdict_rationale: str = ""
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

    # ============================================================
    # Mathematics
    # ============================================================
    "math_algebra": {
        "persona": "algebraist",
        "checklist": [
            "Are the ring / group / module hypotheses (commutative, Noetherian, finitely generated, characteristic) stated at each use?",
            "For category-theoretic constructions: are functor / natural transformation definitions checked on morphisms, not just objects?",
            "Are quotient constructions accompanied by a check that the equivalence relation is compatible (well-defined operations)?",
            "For representation-theoretic claims: is the ground field / characteristic explicit? Are semisimplicity / irreducibility conditions checked?",
            "Are universal properties invoked correctly, with existence AND uniqueness parts addressed?",
            "For homological arguments: are chain complex boundedness / exactness conditions verified?",
            "Are examples given for non-trivial abstract definitions?",
            "Is the distinction between 'up to isomorphism' vs 'equal' respected where it matters?",
        ],
    },
    "math_algebraic_geometry": {
        "persona": "algebraic geometer",
        "checklist": [
            "Are schemes / varieties described with their base field / base scheme explicit?",
            "Is the difference between closed points, geometric points, and generic points respected?",
            "For sheaf-cohomological claims: is the topology (Zariski, étale, fppf) specified? Are coefficient sheaves explicit?",
            "Are properness / separatedness / smoothness hypotheses stated where invoked (base change, GAGA, Serre duality)?",
            "For moduli problems: is the representability / stackiness addressed?",
            "Are singularities characterized (nodal, ADE, log-canonical, terminal) when relevant?",
            "Is the distinction between coarse and fine moduli spaces made where it matters?",
            "For characteristic-p claims: are Frobenius / Verschiebung / crystalline subtleties addressed?",
        ],
    },
    "math_analysis": {
        "persona": "analyst (real / complex / functional / PDE)",
        "checklist": [
            "Are function spaces (Sobolev, Besov, Hölder, L^p) specified with their exponents and domains?",
            "For PDE existence / uniqueness: are boundary conditions and function-space regularity precise?",
            "Are weak / strong / distributional solutions distinguished where the difference matters?",
            "For inequalities: are the constants explicit or are they claimed to depend on which parameters?",
            "Are convergence claims stated in a named topology (a.e., in L^p, weakly, strongly)?",
            "For operator arguments: are domain, closed extension, self-adjointness, and spectrum addressed?",
            "For nonlinear PDE: are blowup / global existence hypotheses matched to the equation's structure?",
            "Are limiting arguments (compactness, weak convergence, extraction) justified rather than asserted?",
        ],
    },
    "math_topology_geometry": {
        "persona": "topologist / differential geometer",
        "checklist": [
            "Are manifold hypotheses (dimension, orientability, compactness, smoothness class) stated?",
            "For (co)homology arguments: is the coefficient ring specified? Are naturality conditions checked?",
            "For fiber bundles / fibrations: are local triviality / lifting conditions explicit?",
            "Are curvature / connection choices (Levi-Civita, principal-connection) named and consistent?",
            "For gauge-theoretic constructions: is the structure group and its action explicit?",
            "For characteristic classes: is the base of the classifying space / bundle addressed?",
            "Are geometric / topological invariants stated with their normalization convention?",
            "For low-dimensional topology: are surgery / handle-decomposition / Heegaard structures given?",
        ],
    },
    "math_combinatorics": {
        "persona": "combinatorist",
        "checklist": [
            "Are graph / hypergraph / poset / matroid definitions given (directed, simple, weighted, labelled)?",
            "For enumeration: are generating function types (ordinary, exponential, Dirichlet) and their formal-vs-analytic status clear?",
            "Are extremal claims accompanied by matching constructions or lower / upper bounds?",
            "For probabilistic method arguments: are independence / concentration hypotheses stated?",
            "Is the difference between 'exact' and 'asymptotic' results respected in statements?",
            "For algebraic combinatorics: is the representation-theoretic or symmetric-function context explicit?",
            "Are algorithmic complexity claims backed by a computational model?",
            "Are recurrences / bijections stated with initial conditions AND correctness argument?",
        ],
    },
    "math_number_theory": {
        "persona": "number theorist",
        "checklist": [
            "For automorphic / L-function arguments: is the level, weight, character, and functional equation stated?",
            "Are analytic estimates (subconvexity, effective bounds) explicit in the constants they depend on?",
            "For arithmetic geometry / Diophantine claims: are heights (Weil, Faltings, canonical) defined with their normalization?",
            "Is the field of definition and its Galois group tracked through constructions?",
            "For p-adic arguments: is the choice of embedding / ramification explicit?",
            "Are conjectural inputs (Riemann hypothesis, BSD, Langlands) flagged as assumptions?",
            "For sieve / analytic arguments: are error terms bounded rather than absorbed silently?",
            "Are ineffective results explicitly labelled as such?",
        ],
    },
    "math_probability": {
        "persona": "probabilist / stochastic analyst",
        "checklist": [
            "Are underlying probability spaces / filtrations / independence structures specified?",
            "For stochastic differential equations: is the SDE solution notion (strong / weak / martingale problem) precise?",
            "Are convergence theorems invoked with their hypotheses (uniform integrability, tightness, Skorohod)?",
            "For random matrix / random graph models: is the model definition unambiguous (edge probability, symmetric matrix ensemble)?",
            "Are large-deviation / concentration statements accompanied by rate functions or explicit constants?",
            "For Markov / mixing arguments: are ergodicity / spectral gap / drift conditions stated?",
            "Is the distinction between quenched vs annealed / in-probability vs almost-sure results maintained?",
            "For couplings / percolation: are monotonicity / stochastic-dominance conditions verified?",
        ],
    },
    "math_logic": {
        "persona": "mathematical logician / foundational theorist",
        "checklist": [
            "Are the underlying theory / language / signature explicit?",
            "For consistency / independence claims: is the base theory (ZFC, ZF, PA, ...) stated?",
            "Are model-theoretic hypotheses (saturation, stability, o-minimality) invoked with their standard meaning?",
            "For forcing / large cardinal arguments: is the type of iteration / cardinal invariant clear?",
            "Are computability / decidability claims stated relative to a specific reduction (Turing, many-one, polynomial)?",
            "For proof-theoretic arguments: is the ordinal / consistency strength calibrated?",
            "Are categoricity / non-standard model results accompanied by explicit examples?",
            "Is the distinction between 'provable' vs 'true in a model' respected?",
        ],
    },

    # ============================================================
    # Chemistry
    # ============================================================
    "chem_organic": {
        "persona": "organic chemist",
        "checklist": [
            "Are synthesis procedures reproducible? (equivalents, temperatures, times, solvents, workup, purification)",
            "For new compounds: are full characterization data (1H/13C NMR, HRMS, IR, mp) reported?",
            "Is the mechanistic proposal supported by experimental evidence (labelling, kinetics, intermediates)?",
            "For stereochemistry claims: is the assignment method explicit (NOE, X-ray, chemical correlation)?",
            "Are yields, dr / ee values, and scope substrate scope reported honestly with substrates that failed?",
            "For catalysis: is the turnover number / frequency reported? Are control experiments (no catalyst, alternative catalyst) shown?",
            "Are reactions checked for reproducibility (multiple runs, different batches)?",
            "Is the novelty compared to prior art with concrete comparisons, not just citation lists?",
        ],
    },
    "chem_inorganic": {
        "persona": "inorganic / organometallic chemist",
        "checklist": [
            "For new complexes: are crystallographic data (CCDC deposition, R-factors, ORTEP) reported?",
            "Is the oxidation state / d-electron count / spin state assignment justified?",
            "For catalytic mechanisms: are the resting state and rate-determining step supported by evidence?",
            "Are ligand-field / MO / DFT arguments consistent with observed magnetism / EPR / UV-vis?",
            "For coordination geometries: are bond lengths / angles compared with literature averages?",
            "Are air- / moisture-sensitive procedures clearly documented (glovebox, Schlenk, drybox)?",
            "For nanomaterials: are size / composition / phase confirmed by multiple techniques (TEM, XRD, XPS)?",
            "Is the difference between 'proposed' and 'demonstrated' mechanism made clear?",
        ],
    },
    "chem_physical": {
        "persona": "physical chemist / spectroscopist",
        "checklist": [
            "Are spectroscopic assignments (IR, Raman, UV-vis, NMR, MS) justified with reference compounds or DFT?",
            "For kinetics: is the rate law derived from data or assumed? Are pseudo-first-order conditions verified?",
            "Are thermodynamic quantities (ΔH, ΔS, ΔG) obtained by van't Hoff / Eyring plots with error analysis?",
            "For time-resolved experiments: are instrument response function, deconvolution, and fit residuals reported?",
            "Are spectroscopic transitions assigned to specific vibrational / electronic modes rather than 'bands'?",
            "For photochemistry: is the quantum yield reported with actinometry?",
            "Are solvent, temperature, and concentration dependencies characterized rather than assumed?",
            "For electrochemistry: are electrodes, reference, scan rates, and cell configuration specified?",
        ],
    },
    "chem_computational": {
        "persona": "computational chemist",
        "checklist": [
            "Are the level of theory (functional, basis set, dispersion correction) and their justification stated?",
            "For DFT: is the functional appropriate for the problem (dispersion, transition states, spin-symmetry breaking)?",
            "Are geometries confirmed as minima / TS by frequency analysis? Are imaginary frequencies characterized?",
            "For solvation: is the model (implicit continuum, explicit, mixed) appropriate?",
            "Are basis-set superposition and completeness effects addressed (counterpoise, extrapolation)?",
            "For excited states: is the method (TDDFT, CASSCF, EOM-CC) appropriate for the state type?",
            "Are benchmarks against experiment or higher-level theory provided?",
            "For MD: are equilibration, sampling, and finite-size effects controlled?",
        ],
    },
    "chem_analytical": {
        "persona": "analytical chemist",
        "checklist": [
            "Are method validation parameters (LOD, LOQ, linearity range, accuracy, precision) reported?",
            "For quantification: are internal standards, calibration curves, and matrix effects addressed?",
            "Are chromatographic conditions (column, gradient, flow, injection) reproducible?",
            "For mass spectrometry: are ionization mode, mass accuracy, and fragmentation patterns reported?",
            "Are sample preparation / extraction recoveries validated?",
            "For trace analysis: are blanks, background subtraction, and interference tests documented?",
            "Are statistical tests appropriate for the sample size (parametric vs non-parametric, replicate design)?",
            "Are reference materials / standards traceable to certified sources?",
        ],
    },
    "chem_biochem_medicinal": {
        "persona": "biochemist / medicinal chemist",
        "checklist": [
            "For enzyme kinetics: are Km, kcat, and mechanism (single/double displacement) fit properly?",
            "Are protein characterization data (purity SDS-PAGE, MW by MS, oligomeric state) reported?",
            "For drug design: is SAR analysis systematic? Are potency, selectivity, and physicochemical properties reported?",
            "Are binding affinity measurements (ITC, SPR, fluorescence) accompanied by control experiments?",
            "For cell / animal assays: is n, replicate independence, and statistical analysis appropriate?",
            "Are compound identity and purity (LC-MS, HPLC) confirmed before biological testing?",
            "For structural biology: are resolution, R-factors, Ramachandran statistics reported?",
            "Are conflict-of-interest / ADME / toxicity implications addressed if drug development is claimed?",
        ],
    },

    # ============================================================
    # Biology
    # ============================================================
    "bio_molecular": {
        "persona": "molecular biologist / geneticist",
        "checklist": [
            "Are cell lines, strains, and constructs authenticated with source and passage / lot info?",
            "For gene manipulation: are guide RNA sequences, homology arms, and off-target checks reported?",
            "Are Western / Northern / Southern blots quantified with loading controls and n>=3 replicates?",
            "For sequencing: are read depth, mapping quality, and variant calling parameters specified?",
            "Are statistical tests appropriate for the data type (Student's t vs Mann-Whitney vs mixed models)?",
            "For knockout / knockdown experiments: are rescue experiments performed?",
            "Are antibody catalog numbers, dilutions, and validation data reported?",
            "Is the difference between 'correlation' and 'causation' respected in the claims?",
        ],
    },
    "bio_cell_dev": {
        "persona": "cell biologist / developmental biologist",
        "checklist": [
            "Are cell types, culture conditions, and passage numbers specified?",
            "For imaging: are microscope, objective, exposure, and image analysis parameters reported?",
            "Are live vs fixed imaging distinctions made? Are photobleaching / phototoxicity controls performed?",
            "For lineage tracing: is the labelling strategy (Cre driver, reporter) unambiguous?",
            "Are developmental staging criteria (embryonic day, somite count, Nieuwkoop / Faber stage) explicit?",
            "For quantification: are cells / embryos independent replicates or technical replicates?",
            "Are morphological claims backed by 3D reconstructions when 2D is insufficient?",
            "For organoid / iPSC work: is the derivation protocol reproducible and validated?",
        ],
    },
    "bio_ecology_evolution": {
        "persona": "ecologist / evolutionary biologist",
        "checklist": [
            "Are sampling design, sample size, and replication levels appropriate for the ecological question?",
            "For phylogenetics: are tree-building method, model selection, and support values (bootstrap, posterior) reported?",
            "Are selection tests (dN/dS, Tajima's D, McDonald-Kreitman) applied with the correct null hypothesis?",
            "For field studies: are environmental covariates, geographic scale, and pseudoreplication addressed?",
            "Are population-genetic estimators (Fst, He, Ne) computed with appropriate models?",
            "For species interactions: is the causal inference (correlation, experiment, natural experiment) clear?",
            "Are climate / disturbance / historical baselines properly established?",
            "Is the study's generalizability beyond the studied system explicit rather than implied?",
        ],
    },
    "bio_neuroscience": {
        "persona": "neuroscientist",
        "checklist": [
            "Are animal / human subject demographics, n, and randomization / blinding described?",
            "For electrophysiology: are recording electrode type, sampling rate, filter, and spike-sorting parameters reported?",
            "For behavior: is task design controlled for confounds (novelty, arousal, motivation, side bias)?",
            "For imaging (fMRI, calcium, EEG): are preprocessing pipeline, atlas, and statistical correction appropriate?",
            "Are neural population claims supported by adequate n neurons / voxels and cross-validation?",
            "For optogenetics / chemogenetics: are targeting specificity and control conditions demonstrated?",
            "Are causal claims from correlational imaging appropriately hedged?",
            "For clinical work: are diagnostic criteria, medication effects, and IRB approval reported?",
        ],
    },
    "bio_bioinformatics_systems": {
        "persona": "bioinformatician / systems biologist",
        "checklist": [
            "Are datasets, versions, and preprocessing pipelines documented reproducibly?",
            "For sequence analysis: are aligner, reference version, and parameter settings specified?",
            "Are batch effects, technical vs biological replicates, and normalization methods appropriate?",
            "For machine learning: are training / validation / test splits reported? Are cross-validation and hyperparameter search described?",
            "Are multiple-testing corrections applied appropriately (FDR, Bonferroni)?",
            "For network / pathway analysis: are gene set enrichment methods and background sets correct?",
            "Is code deposited with version tags? Are dependencies pinned?",
            "For predictive claims: is the negative control / permutation baseline reported?",
        ],
    },
    "bio_biomedicine": {
        "persona": "biomedical / clinical researcher",
        "checklist": [
            "Are sample size, power calculation, and pre-registration status reported?",
            "For clinical / preclinical trials: are randomization, blinding, and inclusion / exclusion criteria explicit?",
            "Are primary vs secondary endpoints declared before analysis?",
            "For diagnostic / prognostic biomarkers: are sensitivity, specificity, AUC, and confidence intervals given?",
            "Are patient demographics, comorbidities, and treatment history balanced across groups?",
            "For epidemiological claims: are confounders identified and adjusted for (DAG, multivariable model)?",
            "Are adverse events, dropouts, and protocol deviations reported?",
            "Is translational relevance (animal model vs human, dose scaling) discussed rather than assumed?",
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

DOMAIN_REVIEWER_PROMPT = """You are a working referee for a scientific journal, with expertise as {persona_description}. You have been asked by the editor to evaluate a manuscript in your field of expertise.

You are given:
- The manuscript itself (attached as PDF)
- The subdomain classification: {subdomain}
- The manuscript is approximately {n_pages} pages long

Your task has two parts. Write each in flowing prose, addressed to the editor, without bullet points or numbered lists.

PART 1: OVERALL STRENGTHS AND WEAKNESSES

First, write the manuscript's overall strengths. Keep this under 200 words. Focus on the substantive contributions: what makes this paper worth publishing, what does it get right technically, what is its most valuable aspect for readers in your field.

Then write the manuscript's overall weaknesses. Keep this under 200 words. Focus on the substantive concerns: what technical or methodological problems does it have, what claims are insufficiently supported, what would you require the authors to address before recommending acceptance.

Separate the two with a clear marker line:

===STRENGTHS===
(your strengths prose here)

===WEAKNESSES===
(your weaknesses prose here)

PART 2: SECTION-BY-SECTION COMMENTARY

After the weaknesses, write another marker line:

===COMMENTARY===

Then write detailed commentary on specific sections or claims of the manuscript. Keep this under 200 words. Reference specific equations, figures, or sections by number where appropriate. This is where you point out technical issues at the level of "in Section 3.2, the derivation of Eq. 14 assumes X, but the authors have not justified this assumption." Write as a technically engaged referee would.

At the very end, on a separate line, write your overall recommendation exactly in one of these formats (pick one):

Domain verdict: accept
Domain verdict: minor revision
Domain verdict: major revision
Domain verdict: reject

Base your verdict on the balance of strengths and weaknesses.

Immediately after the verdict line, on the next line, write a 1-2 sentence rationale for the verdict, prefixed with "Rationale:". For example:

Domain verdict: minor revision
Rationale: The framework is sound and the results are novel, but the restriction to non-spinning primaries and the omission of local drag forces limit immediate astrophysical applicability.

FORMAT REQUIREMENT: Do not write any prose before the "===STRENGTHS===" marker. Start your response with "===STRENGTHS===" on its own line. Do not repeat the strengths content elsewhere; write it once, under the marker.

Use LaTeX notation for any mathematical symbols and equations in your prose. Write `$\\varepsilon$` not `ε`, `$\\mu$` not `µ`, `$\\pi$` not `π`. Every mathematical symbol must be wrapped in `$...$` so the report renders cleanly in markdown with math support."""


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
                        "enum": ["domain_concern"],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "warning"],
                    },
                    "location": {"type": "string"},
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["kind", "severity", "location", "evidence", "confidence"],
            },
        },
    },
    "required": ["findings"],
}


DOMAIN_SYSTEM_TEMPLATE = """You are a senior peer-review referee specialising as a
{persona}. You have been shown a manuscript's title, abstract, section
headings, and introduction. Apply the specialist checklist below to
identify substantive subdomain-specific concerns a generic reviewer
would miss.

SUBDOMAIN CHECKLIST (guidance, not a to-do list; each item does NOT require a finding):
{checklist}

FINDING KIND (only one — domain_concern):
- A substantive subdomain-specific issue: a missing specification,
  unstated assumption, unaddressed regime of validity, ambiguous
  convention, or a central claim whose supporting detail is absent
  from the manuscript summary. What counts as "subdomain-specific"
  is defined by your persona's checklist — not by generic scientific
  writing complaints.

SEVERITY (only two):
- critical: A central claim of the paper rests on the unaddressed issue.
- warning: The referee would ask the author to clarify before publication.

WHEN TO FLAG (subtle — read carefully):

- The paper's title / abstract / section headings CLEARLY IMPLY the
  subdomain concern is CENTRAL to the work, but the summary text gives
  NO evidence the concern is addressed → flag as warning (or critical
  if the whole result depends on it).
- The paper's summary text ADDRESSES a checklist item (even briefly) →
  do NOT flag it.
- The paper's summary is SILENT on a checklist item that is NOT central
  to the paper's aims → do NOT flag it (the item does not apply here).
- You are UNCERTAIN whether a concern is central or peripheral → do NOT
  flag (silence is safer than a speculative warning).

STRICT RULES:

1. Silence is the default. An empty findings list is often the CORRECT
   answer. Do NOT fabricate concerns to fill a quota. Do NOT emit
   generic filler like "the paper could benefit from more discussion".

2. Evidence must be a SHORT direct quote or a specific reference to
   the summary text provided. Do NOT paraphrase — quote the exact
   phrase (or its absence) that concerns you.

3. Location must be specific: name the section, sentence, or equation.
   Do not use vague locations like "Throughout" or "General".

4. If the paper falls outside your subdomain expertise (mismatched
   routing), return an empty findings list — no filler note.

5. Confidence must be >= 0.5. Below that, omit the finding entirely.

Target: 0-4 findings. Empty is common and correct. Never exceed 5."""


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


def _dedupe_findings(findings: list[DomainFinding]) -> list[DomainFinding]:
    """Same (location, kind) collapses to one finding. Keeps first occurrence."""
    seen: set[tuple[str, str]] = set()
    out = []
    for f in findings:
        key = ((f.location or "").strip().lower(), f.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def analyze_domain(doi: str, client: Optional[GeminiClient] = None, uploaded_file: Any = None) -> DomainReport:
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

    # Load persona description (from DOMAIN_PROMPTS or DEFAULT_UNIVERSAL).
    if subdomain in DOMAIN_PROMPTS:
        persona = DOMAIN_PROMPTS[subdomain]["persona"]
    else:
        persona = DEFAULT_UNIVERSAL["persona"]
    report.persona = persona

    # Compute page-count-adaptive paragraph counts.
    n_pages = parsed.get("n_pages", 0) or 10
    if n_pages < 10:
        n_strengths_para = 1
        n_weaknesses_para = 1
        n_commentary_para = 1
    elif n_pages < 20:
        n_strengths_para = 2
        n_weaknesses_para = 2
        n_commentary_para = 2
    elif n_pages < 30:
        n_strengths_para = 2
        n_weaknesses_para = 2
        n_commentary_para = 3
    else:
        n_strengths_para = 3
        n_weaknesses_para = 3
        n_commentary_para = 4

    system_prompt = DOMAIN_REVIEWER_PROMPT.format(
        persona_description=persona,
        subdomain=subdomain,
        n_pages=n_pages,
    )

    # PDF handling: prefer uploaded_file (cached), fall back to inline bytes.
    pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    pdf_bytes = None
    if uploaded_file is None:
        pdf_bytes = pdf_path.read_bytes() if pdf_path.exists() else None

    user_prose = f"Manuscript subdomain: {subdomain}. Approximate length: {n_pages} pages.\n\nPlease produce the two-part evaluation as instructed."

    model_name = CONFIG["gemini"]["model_specialist"]
    text, in_tok, out_tok = client.plain_text(
        model=model_name,
        system=system_prompt,
        user=user_prose,
        max_output_tokens=4000,
        temperature=0.3,
        pdf_bytes=pdf_bytes,
        uploaded_file=uploaded_file,
    )

    # Parse the marker-delimited output with preamble fallback.
    # Gemini occasionally skips the ===STRENGTHS=== marker and writes the
    # strengths content as an unmarked preamble. We collect preamble text
    # into strengths by default; if ===STRENGTHS=== is encountered, that
    # marked content overrides the preamble. If a subsequent marker is
    # encountered without any preceding strengths marker, the preamble is
    # committed to strengths.
    strengths = ""
    weaknesses = ""
    commentary = ""
    verdict = ""
    verdict_rationale = ""

    current_section = "preamble"
    seen_strengths_marker = False
    buf: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("===STRENGTHS==="):
            # Discard preamble content since the model gave us an explicit
            # strengths section.
            current_section = "strengths"
            seen_strengths_marker = True
            buf = []
            continue
        if stripped.startswith("===WEAKNESSES==="):
            content = "\n".join(buf).strip()
            if current_section in ("preamble", "strengths"):
                # If we never saw ===STRENGTHS===, promote preamble to strengths.
                strengths = content
            current_section = "weaknesses"
            buf = []
            continue
        if stripped.startswith("===COMMENTARY==="):
            content = "\n".join(buf).strip()
            if current_section == "weaknesses":
                weaknesses = content
            elif current_section in ("preamble", "strengths") and not seen_strengths_marker:
                strengths = content
            current_section = "commentary"
            buf = []
            continue
        if stripped.lower().startswith("domain verdict:"):
            content = "\n".join(buf).strip()
            if current_section == "commentary":
                commentary = content
            elif current_section == "weaknesses":
                weaknesses = content
            elif current_section in ("preamble", "strengths") and not seen_strengths_marker:
                strengths = content
            verdict = stripped.split(":", 1)[1].strip().lower()
            current_section = "verdict_rationale"
            buf = []
            continue
        if stripped.lower().startswith("rationale:") and current_section == "verdict_rationale":
            verdict_rationale = stripped.split(":", 1)[1].strip()
            # Keep accumulating in case rationale spans multiple lines below.
            buf = []
            current_section = "verdict_rationale_continued"
            continue
        buf.append(line)

    # Flush any remaining buffer to the appropriate section.
    trailing = "\n".join(buf).strip()
    if trailing:
        if current_section == "verdict_rationale_continued":
            # Continuation of rationale on subsequent lines.
            verdict_rationale = (verdict_rationale + " " + trailing).strip() if verdict_rationale else trailing
        elif current_section == "commentary" and not commentary:
            commentary = trailing
        elif current_section == "weaknesses" and not weaknesses:
            weaknesses = trailing
        elif current_section in ("preamble", "strengths") and not seen_strengths_marker and not strengths:
            strengths = trailing

    report.strengths_narrative = strengths
    report.weaknesses_narrative = weaknesses
    report.commentary_narrative = commentary
    report.verdict = verdict
    report.verdict_rationale = verdict_rationale

    report.findings = []
    report.n_findings = 0
    report.n_critical = 0
    report.n_warning = 0
    report.model = model_name
    report.input_tokens = in_tok
    report.output_tokens = out_tok
    return report


def save_report(report: DomainReport, out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "domain_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
