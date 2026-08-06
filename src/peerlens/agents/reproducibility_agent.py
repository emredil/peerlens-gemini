"""Reproducibility Agent (gate-based).

Design principles:
  1. If the paper is theoretical AND no signals (no code URLs, no
     algorithm mentions, no availability statements) exist in the text,
     skip the Gemini call entirely and return an empty report. Pure
     theory papers get zero reproducibility findings by design.
  2. Only 'warning' and 'critical' severities are allowed. No 'info'.
  3. Only concrete gap kinds are emitted: missing_code, missing_data,
     missing_params, missing_algorithm. No positive_signal, no info.
     If nothing concrete is missing, return an empty findings list.

The goal is that this agent produces zero output unless it has a real,
actionable reproducibility concern that a rigorous referee would raise.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


@dataclass
class ReproFinding:
    kind: str          # "missing_code" | "missing_data" | "missing_params" | "missing_algorithm"
    severity: str      # "critical" | "warning"
    location: str
    evidence: str
    suggested_fix: str = ""
    confidence: float = 0.0


@dataclass
class ReproReport:
    doi_label: str
    paper_types: list[str] = field(default_factory=list)
    scrutiny_level: str = "none"  # none | medium | deep
    narrative: str = ""
    verdict: str = ""  # "reproducible" | "mostly reproducible" | "gaps" | "not applicable" 
    detected_signals: dict = field(default_factory=dict)
    gate_reason: str = ""
    n_findings: int = 0
    n_critical: int = 0
    n_warning: int = 0
    findings: list[ReproFinding] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


CODE_URL_PATTERNS = [
    re.compile(r"https?://(?:www\.)?github\.com/[\w\-/.]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?gitlab\.com/[\w\-/.]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?bitbucket\.org/[\w\-/.]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?zenodo\.org/[\w\-/.]+", re.IGNORECASE),
    re.compile(r"https?://doi\.org/10\.5281/[\w.-]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?figshare\.com/[\w\-/.]+", re.IGNORECASE),
]

AVAILABILITY_KEYWORDS = [
    "code available", "code is available", "source code",
    "data available", "data is available", "data availability",
    "reproducibility statement", "computational reproducibility",
    "supplementary code", "publicly available", "open-source",
]

BIO_AVAILABILITY_KEYWORDS = [
    "code available", "code is available", "data available",
    "data is available", "supplementary material", "supplementary data",
    "github.com", "zenodo", "figshare",
    "GEO accession", "SRA accession", "ENA accession", "BioProject",
    "PDB code", "UniProt", "ArrayExpress", "PRIDE", "MassIVE",
    "dryad", "OSF", "protocols.io", "IRB approval", "ethics approval",
    "IACUC", "animal protocol", "institutional review",
    "sample size", "n =", "n=", "replicates", "biological replicate",
    "technical replicate", "randomization", "blinding",
]

BIO_METHODOLOGY_KEYWORDS = [
    "cell line", "primary cells", "iPSC", "stem cell", "organoid",
    "knockout", "knockdown", "CRISPR", "Cas9", "guide RNA", "sgRNA",
    "antibody", "clone", "catalog", "monoclonal", "polyclonal",
    "PCR", "qPCR", "RT-PCR", "sequencing", "RNA-seq", "scRNA-seq",
    "ChIP-seq", "ATAC-seq", "Western blot", "immunoblot",
    "flow cytometry", "FACS", "immunofluorescence", "microscopy",
    "confocal", "two-photon", "electrophysiology", "patch clamp",
    "optogenetics", "chemogenetics", "DREADD",
    "phylogenetic", "bootstrap", "posterior probability",
    "differential expression", "GSEA", "enrichment analysis",
]

CHEM_AVAILABILITY_KEYWORDS = [
    "supporting information", "SI", "supplementary information",
    "characterization data", "NMR spectra", "mass spectra",
    "crystallographic data", "CCDC deposition", "CIF file",
    "raw spectra", "raw data", "figshare", "zenodo", "OSF",
    "code available", "github.com",
]

CHEM_METHODOLOGY_KEYWORDS = [
    "HPLC", "LC-MS", "GC-MS", "NMR", "1H NMR", "13C NMR",
    "IR spectroscopy", "FTIR", "Raman", "UV-Vis", "UV-visible",
    "XRD", "XPS", "EXAFS", "EELS", "TEM", "SEM", "HAADF-STEM",
    "EDS", "ICP-MS", "TGA", "DSC", "BET", "DLS",
    "cyclic voltammetry", "EIS", "impedance spectroscopy",
    "DFT", "density functional theory", "ab initio",
    "molecular dynamics", "MD simulation", "force field",
    "synthesis", "purification", "column chromatography",
    "recrystallization", "workup", "yield",
    "characterization", "titration", "calibration curve",
]


ALGORITHM_KEYWORDS = [
    "monte carlo", "MCMC", "tensor network", "DMRG", "MPS",
    "matrix product", "quantum monte carlo",
    "molecular dynamics", "lattice qcd", "finite element",
    "runge-kutta", "verlet", "exact diagonalization",
    "hartree-fock", "density functional", "DFT",
    "path integral", "worldline", "lattice regularization",
    "numerical simulation", "numerical solver",
]


def _scan_signals(parsed: dict, field: str = "physics") -> dict:
    text_parts = [parsed.get("title", ""), parsed.get("abstract", "")]
    for s in parsed.get("sections", []):
        text_parts.append(s.get("text", ""))
    for r in parsed.get("references", []):
        text_parts.append(r)
    full_text = "\n".join(text_parts)
    full_text_lower = full_text.lower()

    urls = []
    for pat in CODE_URL_PATTERNS:
        for m in pat.finditer(full_text):
            url = m.group(0)
            if url not in urls:
                urls.append(url)

    # Pick field-specific keyword lists. Physics uses the originals.
    if field == "biology":
        avail_list = AVAILABILITY_KEYWORDS + BIO_AVAILABILITY_KEYWORDS
        algo_list = ALGORITHM_KEYWORDS + BIO_METHODOLOGY_KEYWORDS
    elif field == "chemistry":
        avail_list = AVAILABILITY_KEYWORDS + CHEM_AVAILABILITY_KEYWORDS
        algo_list = ALGORITHM_KEYWORDS + CHEM_METHODOLOGY_KEYWORDS
    else:
        avail_list = AVAILABILITY_KEYWORDS
        algo_list = ALGORITHM_KEYWORDS

    availability = [kw for kw in avail_list if kw.lower() in full_text_lower]
    algorithms = [kw for kw in algo_list if kw.lower() in full_text_lower]

    return {
        "code_urls": urls,
        "availability_statements": availability,
        "algorithms_mentioned": algorithms,
        "n_tables": len(parsed.get("table_captions", [])),
        "n_figures": len(parsed.get("figure_captions", [])),
    }


def _classification_for(doi: str) -> Optional[dict]:
    path = get_path("processed_data") / "classifications" / f"{doi}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _decide_gate(paper_types: list[str], signals: dict, field: str = "physics") -> tuple[str, str]:
    """Return (scrutiny_level, gate_reason).

    Levels:
      none   : skip Gemini entirely, empty report
      medium : some signals present but paper is nominally theoretical
      deep   : paper_type explicitly computational / experimental /
               phenomenological, or multiple strong signals present
    """
    # Field-conditional early routing (checked BEFORE physics-specific logic).
    if field == "math":
        return "none", "field=math: reproducibility axis (code/data/protocol) does not apply"

    if field == "biology":
        if any(t in ("experimental", "computational") for t in paper_types):
            return "deep", "field=biology: experimental/computational paper needs deep scrutiny (cell lines, antibodies, replicate design, code/data deposit)"
        return "medium", "field=biology: non-experimental paper, medium scrutiny"

    if field == "chemistry":
        if any(t in ("experimental", "computational") for t in paper_types):
            return "deep", "field=chemistry: experimental/computational paper needs deep scrutiny (synthesis, characterization, validation detail)"
        return "medium", "field=chemistry: non-experimental paper, medium scrutiny"

    # Physics: original logic below.
    heavy = {"computational", "experimental", "phenomenological"}
    is_heavy = bool(paper_types) and any(t in heavy for t in paper_types)

    n_urls = len(signals.get("code_urls", []))
    n_avail = len(signals.get("availability_statements", []))
    n_algo = len(signals.get("algorithms_mentioned", []))

    strong = 0
    if n_urls > 0: strong += 1
    if n_avail > 0: strong += 1
    if n_algo >= 2: strong += 1

    if is_heavy or strong >= 2:
        return "deep", f"heavy paper_type={is_heavy}, strong_signals={strong}"

    if strong >= 1:
        return "medium", f"1 strong signal (urls={n_urls}, avail={n_avail}, algos={n_algo})"

    if n_algo >= 1:
        return "medium", f"single algorithm mention: {signals.get('algorithms_mentioned')}"

    return "none", "no signals detected, paper appears purely theoretical"


REPRO_REVIEWER_PROMPT = """You are a working referee for a scientific journal. You have been asked by the editor to evaluate the reproducibility of a manuscript.

You are given:
- The manuscript itself (attached as PDF)
- The manuscript's field: {field}
- The manuscript's subdomain classification: {subdomain}
- The manuscript is approximately {n_pages} pages long
- The paper types: {paper_types} (for example, theoretical, experimental, computational)

Your task is to write prose commentary on the manuscript's reproducibility, addressed to the editor. Keep this under 200 words. No bullet points, no numbered lists.

Consider the following as guidance, but only write about points that genuinely apply to this manuscript in its field:

- Code availability: does the manuscript provide a code repository (GitHub, Zenodo, or similar)? Is the code specific enough that another group could rerun the analysis? If code is essential to the paper (numerical work, machine learning, simulations), is it deposited?
- Data availability: for experimental or observational papers, are the underlying datasets provided? For chemistry papers, is supporting information (SI) with characterization data available? For biology papers, are datasets deposited (GEO, SRA, PDB, or similar)? For computational papers, are inputs and outputs available?
- Method reporting: are experimental protocols described in enough detail that a competent lab could reproduce them? For simulations, are parameters, initial conditions, and software versions specified? For theoretical papers, are derivations complete enough that a reader can follow every step?
- Field-specific concerns: for chemistry, HPLC/NMR/XRD conditions and purity specifications; for biology, cell lines, antibody catalog numbers, replicate counts, sample sizes and their justification, and IRB/IACUC approvals where relevant; for physics experimental, instrument specifications and calibration.

Focus on substantive reproducibility issues that would prevent an independent group from reproducing the work. If the paper is a pure theoretical or mathematics work where these considerations do not apply, write a single brief sentence stating so and stop.

At the end, on a separate line, write your overall recommendation exactly in one of these formats (pick one):

Reproducibility verdict: reproducible
Reproducibility verdict: mostly reproducible
Reproducibility verdict: gaps
Reproducibility verdict: not applicable

Use "not applicable" only for pure math or pure theory papers where reproducibility is not a meaningful axis.

Use LaTeX notation for any mathematical symbols and equations in your prose. Write `$\\varepsilon$` not `ε`, `$\\mu$` not `µ`, `$\\pi$` not `π`. Every mathematical symbol must be wrapped in `$...$` so the report renders cleanly in markdown with math support."""


REPRO_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["missing_code", "missing_data",
                                 "missing_params", "missing_algorithm"],
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


SYSTEM_TEMPLATES = {
    "deep": """You are a physics referee focused on REPRODUCIBILITY of a computational,
experimental, or phenomenological paper.

Emit findings ONLY for CONCRETE, ACTIONABLE reproducibility GAPS. A gap is
worth flagging only if a rigorous referee would ask the author to add
specific information before publication.

Allowed finding kinds (no others exist):
  - missing_code:      Numerical or computational results are shown but
                       no code repository is cited or made available.
  - missing_data:      Data used to generate figures / tables is not
                       accompanied by a data availability statement.
  - missing_params:    A simulation or numerical calculation is described
                       but specific parameter values (system size,
                       coupling constants, seeds, iteration counts,
                       precision) are not fully reported.
  - missing_algorithm: A method is invoked (Monte Carlo, DMRG, exact
                       diagonalization, DFT, ...) without specifying
                       which library, variant, or version.

Severity: 'warning' unless the paper's central result rests on the
un-shared / unspecified item, in which case 'critical'.

STRICT RULES:
  1. Do NOT emit findings for good practices, positive observations,
     acknowledgments, or descriptive notes. If the paper does something
     well, stay silent — positive notes are not your job.
  2. Do NOT emit findings that only paraphrase what the paper already
     states elsewhere.
  3. If NOTHING concrete is missing, return an EMPTY findings list.
     Empty is the correct answer whenever there is no gap.

Target: 0 to 4 findings. Silence is preferable to filler.""",

    "medium": """You are a physics referee checking reproducibility. The paper is
nominally theoretical but has some computational or numerical component.

Emit findings ONLY for concrete missing information a rigorous referee
would demand before publication:
  - missing_code:      Numerics shown, code not shared.
  - missing_params:    Specific numerical values / plots without the
                       parameters that generated them.

Allowed severities: 'warning' or 'critical' only.

STRICT RULES:
  1. Do NOT emit findings for good practices or acknowledgments.
  2. Do NOT emit findings that duplicate paper content.
  3. Empty findings list is the correct answer whenever nothing
     concrete is missing.

Target: 0 to 2 findings. Silence is the default.""",
}


def _format_paper_context(parsed: dict, signals: dict) -> str:
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

REPRODUCIBILITY SIGNALS DETECTED (from pattern scan):
  Code / data URLs found: {signals['code_urls'] or 'NONE'}
  Availability keyword hits: {signals['availability_statements'] or 'NONE'}
  Algorithm mentions: {signals['algorithms_mentioned'] or 'NONE'}
  Tables: {signals['n_tables']}, Figures: {signals['n_figures']}"""


def analyze_reproducibility(doi: str, client: Optional[GeminiClient] = None, uploaded_file: Any = None) -> ReproReport:
    report = ReproReport(doi_label=doi)

    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not parsed_path.exists():
        report.gate_reason = "parsed JSON missing"
        return report
    parsed = json.loads(parsed_path.read_text())

    classification = _classification_for(doi)
    paper_types = (classification or {}).get("paper_types", []) or []
    report.paper_types = paper_types

    field_top = (classification or {}).get("field_top", "physics")

    signals = _scan_signals(parsed, field=field_top)
    report.detected_signals = signals

    scrutiny, reason = _decide_gate(paper_types, signals, field=field_top)
    report.scrutiny_level = scrutiny
    report.gate_reason = reason

    # GATE: if no scrutiny needed, skip Gemini entirely
    if scrutiny == "none":
        return report

    if client is None:
        client = GeminiClient()

    # Determine paragraph budget from paper length.
    n_pages = parsed.get("n_pages", 0) or 10
    if n_pages < 10:
        n_para = 1
    elif n_pages < 25:
        n_para = 1
    else:
        n_para = 2

    subdomain = ""
    subdomains_list = (classification or {}).get("subdomains", [])
    if subdomains_list:
        subdomain = subdomains_list[0]

    system_prompt = REPRO_REVIEWER_PROMPT.format(
        field=field_top,
        subdomain=subdomain or "unknown",
        n_pages=n_pages,
        paper_types=", ".join(paper_types) if paper_types else "unspecified",
    )

    # PDF handling: prefer uploaded_file (cached), fall back to inline bytes.
    pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    pdf_bytes = None
    if uploaded_file is None:
        pdf_bytes = pdf_path.read_bytes() if pdf_path.exists() else None

    user_prose = (
        f"Manuscript field: {field_top}. Subdomain: {subdomain or 'unknown'}. "
        f"Paper types: {', '.join(paper_types) if paper_types else 'unspecified'}. "
        f"Approximate length: {n_pages} pages. "
        f"Scrutiny level assigned by pipeline: {scrutiny}.\n\n"
        f"Please produce the reproducibility commentary as instructed."
    )

    model_name = CONFIG["gemini"]["model_specialist"]
    text, in_tok, out_tok = client.plain_text(
        model=model_name,
        system=system_prompt,
        user=user_prose,
        max_output_tokens=8000,
        temperature=0.3,
        pdf_bytes=pdf_bytes,
        uploaded_file=uploaded_file,
    )

    # Parse prose + verdict.
    narrative_lines = []
    verdict = ""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("reproducibility verdict:"):
            verdict = stripped.split(":", 1)[1].strip().lower()
        else:
            narrative_lines.append(line)
    report.narrative = "\n".join(narrative_lines).strip()
    report.verdict = verdict

    report.findings = []
    report.n_findings = 0
    report.n_critical = 0
    report.n_warning = 0
    report.model = model_name
    report.input_tokens = in_tok
    report.output_tokens = out_tok
    return report


def save_report(report: ReproReport, out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "reproducibility_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
