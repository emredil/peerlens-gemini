"""Claim Extractor (Gemini port): text-only agent that identifies checkable math claims."""
from __future__ import annotations

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
- Prefer algebraic_identity over symmetry for concrete commutation statements."""


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


def extract_claims(parsed_paper: dict, client: GeminiClient | None = None) -> ExtractedClaims:
    if client is None:
        client = GeminiClient()
    model = CONFIG["gemini"]["model_classifier"]

    result = client.structured_call(
        model=model,
        system=SYSTEM_PROMPT,
        user=f"Extract math claims from this paper.\n\n{_format_paper(parsed_paper)}",
        tool_name="record_math_claims",
        tool_description="Record all verifiable mathematical claims AND catalogue at least 5 key symbols.",
        tool_input_schema=CLAIM_SCHEMA,
        max_output_tokens=8192,
    )

    payload = result.tool_input
    valid_types = {"dimensional", "limit", "reduction", "algebraic_identity", "symmetry", "conservation"}
    for c in payload.get("claims", []):
        if c.get("claim_type") not in valid_types:
            c["claim_type"] = "algebraic_identity"

    claims = [MathClaim(
        claim_type=c["claim_type"],
        location=c.get("location", ""),
        referenced_equations=c.get("referenced_equations", []),
        natural_language=c["natural_language"],
        check_hint=c["check_hint"],
        verification_method=c.get("verification_method", "manual"),
    ) for c in payload.get("claims", [])]

    symbols = [SymbolDefinition(
        symbol=s["symbol"],
        meaning=s["meaning"],
        units=s["units"],
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
