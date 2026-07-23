"""Classifier agent (Gemini port): subdomain + paper type via structured output."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from peerlens.classification.taxonomy import (
    SUBDOMAIN_CODES, SUBDOMAIN_DESCRIPTIONS, PAPER_TYPES,
)
from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


FEW_SHOT_DOIS = [
    "SciPostPhys.20.5.138",
    "SciPostPhys.20.3.079",
    "SciPostPhys.20.6.161",
]


@dataclass
class ClassificationResult:
    doi_label: str
    subdomains: list[str] = field(default_factory=list)
    paper_types: list[str] = field(default_factory=list)
    subdomain_confidence: dict[str, float] = field(default_factory=dict)
    paper_type_confidence: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def _build_system_prompt() -> str:
    subdomain_list = "\n".join(
        f"  - {code}: {SUBDOMAIN_DESCRIPTIONS[code]}"
        for code in SUBDOMAIN_CODES
    )
    return f"""You are a physics manuscript classifier trained to mimic SciPost Physics editorial decisions.

Given a paper's title, abstract, section headings, and introduction snippet,
assign:
  1. SUBDOMAIN codes from the controlled vocabulary below (multi-label).
  2. PAPER TYPE codes from: {", ".join(PAPER_TYPES)}.

SUBDOMAIN vocabulary:
{subdomain_list}

CRITICAL LABELLING PRINCIPLES:

1. Be MINIMAL. SciPost editors typically assign 1-2 specialties, sometimes just
   1. Assign a subdomain only if it is a *primary* focus of the paper. Do NOT
   tag every topic the paper touches.

2. `quantum_physics` is reserved for quantum information, quantum computing,
   quantum protocols, quantum measurement theory. NOT a catch-all for any
   paper mentioning quantum mechanics.

3. `math_physics` is a secondary label. Add it when the central contribution
   is a mathematical result applied to a physics system.

4. PAPER TYPE criteria:
   - `theoretical`: derivations, models, analytical results, proofs.
   - `computational`: numerical simulations or heavy data-driven analysis.
   - `experimental`: new measurements or apparatus. Rare in SciPost Physics.
   - `phenomenological`: bridging theory with observable signatures,
     typically HEP. Often combined with `computational` for MC simulations.

5. Prefer accuracy over coverage. Omit uncertain tags."""


CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "subdomains": {
            "type": "array",
            "items": {"type": "string", "enum": SUBDOMAIN_CODES},
            "minItems": 1,
        },
        "paper_types": {
            "type": "array",
            "items": {"type": "string", "enum": PAPER_TYPES},
            "minItems": 1,
        },
        "rationale": {"type": "string"},
    },
    "required": ["subdomains", "paper_types", "rationale"],
}


def _format_paper(parsed: dict) -> str:
    sec_headings = "\n".join(
        f"  {s['number']} {s['title']}"
        for s in parsed.get("sections", [])[:20]
    )
    intro_snippet = ""
    if parsed.get("sections"):
        intro_snippet = parsed["sections"][0]["text"][:3000]

    return f"""TITLE:
{parsed.get('title', '')}

ABSTRACT:
{parsed.get('abstract', '')}

SECTION HEADINGS:
{sec_headings}

INTRODUCTION SNIPPET:
{intro_snippet}

EQUATION COUNT: {parsed.get('equation_count', 0)}
FIGURE COUNT: {len(parsed.get('figure_captions', []))}
TABLE COUNT: {len(parsed.get('table_captions', []))}
PAGE COUNT: {parsed.get('n_pages', 0)}"""


_FEW_SHOT_CACHE: list[dict] | None = None


def _load_few_shot_examples() -> list[dict]:
    import ast
    from peerlens.classification.taxonomy import normalize_specialties

    parsed_dir = get_path("processed_data") / "parsed"
    meta_dir = get_path("raw_data") / "metadata"

    meta_lookup = {}
    for f in meta_dir.glob("SciPostPhys*.jsonl"):
        for line in f.open():
            r = json.loads(line)
            meta_lookup[r["doi_label"]] = r

    examples = []
    for doi in FEW_SHOT_DOIS:
        pf = parsed_dir / f"{doi}.json"
        if not pf.exists() or doi not in meta_lookup:
            continue
        parsed = json.loads(pf.read_text())
        meta = meta_lookup[doi]
        try:
            apps = ast.literal_eval(meta["approaches"])
        except Exception:
            apps = []
        examples.append({
            "parsed": parsed,
            "subdomains": normalize_specialties(meta["specialties"]),
            "paper_types": [a for a in apps if a in PAPER_TYPES],
        })
    return examples


def _build_user_prompt(parsed_paper: dict, few_shots: list[dict]) -> str:
    """Gemini doesn't accept synthetic multi-turn tool_use history as cleanly
    as Anthropic, so we inline the few-shot examples as prose in the user
    message."""
    parts = []
    parts.append("Below are three worked examples of SciPost editorial decisions, "
                 "followed by the paper you must classify. Learn from the examples, "
                 "then call the classifier for the final paper.\n")
    for i, ex in enumerate(few_shots, 1):
        parts.append(f"--- EXAMPLE {i} ---")
        parts.append(_format_paper(ex["parsed"]))
        parts.append(f"Correct classification: subdomains={ex['subdomains']}, "
                     f"paper_types={ex['paper_types']}")
        parts.append("")
    parts.append("--- PAPER TO CLASSIFY ---")
    parts.append(_format_paper(parsed_paper))
    return "\n".join(parts)


def classify_paper(parsed_paper: dict, client: GeminiClient | None = None) -> ClassificationResult:
    global _FEW_SHOT_CACHE
    if _FEW_SHOT_CACHE is None:
        _FEW_SHOT_CACHE = _load_few_shot_examples()
    if client is None:
        client = GeminiClient()

    model = CONFIG["gemini"]["model_classifier"]
    user = _build_user_prompt(parsed_paper, _FEW_SHOT_CACHE)

    result = client.structured_call(
        model=model,
        system=_build_system_prompt(),
        user=user,
        tool_name="classify_paper",
        tool_description="Return the subdomain(s) and paper type(s) for the paper.",
        tool_input_schema=CLASSIFY_SCHEMA,
        max_output_tokens=1024,
    )

    payload = result.tool_input
    return ClassificationResult(
        doi_label=parsed_paper["doi_label"],
        subdomains=payload.get("subdomains", []),
        paper_types=payload.get("paper_types", []),
        subdomain_confidence=payload.get("subdomain_confidence", {}) or {},
        paper_type_confidence=payload.get("paper_type_confidence", {}) or {},
        rationale=payload.get("rationale", ""),
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def save_classification(result: ClassificationResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    return out_path
