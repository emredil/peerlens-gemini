"""LLM detector — opt-in policy check for AI-generated manuscripts.

Design
------
- Opt-in: only runs when the editor explicitly requests it (--check-llm
  or the web UI checkbox). Journals that legalize / encourage LLM-drafted
  submissions leave this off.
- Never contributes to the triage score. This is a policy check, not
  a quality signal. Editor decides separately whether LLM authorship
  disqualifies the paper under their journal's policy.
- Bias to `unclear`: false positives smear real authors' reputations,
  so the model abstains when signals are weak or mixed.
- Uses gemini-flash-latest (configured as model_llm_detector) — this
  is pattern recognition, not deep reasoning; pro is overkill.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


MODEL_NAME = CONFIG["gemini"].get("model_llm_detector", "gemini-flash-latest")


Verdict = str  # "likely_human" | "unclear" | "possibly_llm" | "likely_llm"


@dataclass
class LlmDetectionReport:
    doi_label: str
    verdict: Verdict = "unclear"
    confidence: float = 0.0
    signals: list[dict] = field(default_factory=list)
    reasoning: str = ""
    model: str = ""
    n_signals: int = 0
    n_strong: int = 0
    n_moderate: int = 0
    n_weak: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


SYSTEM_PROMPT = """You are an academic integrity officer specialising in
detecting LLM-generated scientific papers.

You have exactly one job: read this manuscript and judge whether it was
written by a human physicist or generated (fully or largely) by an LLM.

BE CONSERVATIVE. False positives smear real authors' reputations. When
signals are weak, mixed, or ambiguous, respond with `unclear`. Only
respond `likely_llm` when there is clear cumulative evidence across
multiple categories. `possibly_llm` requires at least two moderate
signals in different categories.

Signal categories to examine:

1. STYLOMETRIC
   - Uniform sentence length; low variance in paragraph structure
   - Over-use of discourse markers ("moreover", "furthermore",
     "in addition", "notably", "importantly", "it is worth noting")
   - Formulaic transitions between paragraphs
   - Absence of typos, informal phrasing, personal voice
   - Symmetrical "on the one hand / on the other hand" constructions
   - Absence of natural hedging that domain experts use

2. CONTENT
   - Fabricated-looking citations (unusual DOI patterns, mismatched
     author/title/venue, references that do not match the claim being
     supported)
   - Textbook-recitation of well-known material with no original synthesis
   - Absence of specific technical details a practitioner would give
     (equipment settings, exact conventions, gauge choices, code versions)
   - No idiosyncratic observations, anecdotes, or asides
   - Vague hand-waving instead of concrete physical arguments

3. STRUCTURAL
   - Every section follows the same rhetorical pattern
   - Motivations expressed as enumerated bullet-like clauses rather
     than physical intuition
   - Uniform paragraph lengths across the paper
   - Introduction and conclusion mirror each other section-by-section
   - Excessive signposting ("In this paper we will show that...")

4. PHYSICS-SPECIFIC
   - Formulas that appear without derivation reasoning or citation
   - Symbols defined but never used, or used but never defined
   - "This is important because..." templated justifications
   - Hidden contradictions across sections (a claim in one section,
     the opposite quietly assumed in another)
   - Hand-wavy physical arguments that would not survive real
     collaborator questioning

VERDICT RULES:
  likely_human   : zero or one weak signal; nothing across categories
  unclear        : one or two signals but not cumulative; ambiguous case
  possibly_llm   : >=2 moderate signals across >=2 categories
  likely_llm     : strong cumulative evidence across >=3 categories,
                   or >=2 strong signals in different categories

If you find no meaningful signals: empty signals list, verdict
`likely_human`, one-sentence reasoning. Do NOT invent signals just to
fill the list. Do NOT flag surface polish alone as evidence."""


LLM_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["likely_human", "unclear", "possibly_llm", "likely_llm"],
        },
        "confidence": {"type": "number"},
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["stylometric", "content",
                                 "structural", "physics_specific"],
                    },
                    "description": {"type": "string"},
                    "location": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["weak", "moderate", "strong"],
                    },
                },
                "required": ["category", "description", "severity"],
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "confidence", "signals", "reasoning"],
}


def _format_paper_for_detection(parsed: dict) -> str:
    """Compact string form of the paper for the model."""
    lines = []
    lines.append(f"# Title: {parsed.get('title', '')}")
    if parsed.get("authors"):
        auths = parsed["authors"]
        if isinstance(auths, list):
            lines.append(f"Authors: {', '.join(str(a) for a in auths[:10])}")
        else:
            lines.append(f"Authors: {auths}")
    lines.append("")
    if parsed.get("abstract"):
        lines.append("## Abstract")
        lines.append(parsed["abstract"])
        lines.append("")

    for s in parsed.get("sections", [])[:40]:
        if isinstance(s, dict):
            title = s.get("title") or s.get("heading") or ""
            text = s.get("text") or s.get("body") or s.get("content") or ""
        else:
            title = ""
            text = str(s)
        lines.append(f"## {title}")
        lines.append(text[:4000])
        lines.append("")

    refs = parsed.get("references", [])
    if refs:
        lines.append("## References")
        for i, r in enumerate(refs[:60], 1):
            if isinstance(r, dict):
                lines.append(f"[{i}] {r.get('raw') or r.get('title') or ''}")
            elif isinstance(r, str):
                lines.append(f"[{i}] {r}")
            else:
                lines.append(f"[{i}] {r}")
    return "\n".join(lines)


def analyze_llm_authorship(
    doi_label: str, *, client: Optional[GeminiClient] = None
) -> LlmDetectionReport:
    parsed_path = get_path("processed_data") / "parsed" / f"{doi_label}.json"
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed paper missing: {parsed_path}")
    parsed = json.loads(parsed_path.read_text())

    if client is None:
        client = GeminiClient()

    paper_text = _format_paper_for_detection(parsed)

    result = client.structured_call(
        model=MODEL_NAME,
        system=SYSTEM_PROMPT,
        user=f"Analyse this manuscript for LLM authorship signals:\n\n{paper_text}",
        tool_name="record_llm_detection",
        tool_description="Record the LLM-authorship verdict and signals.",
        tool_input_schema=LLM_DETECTION_SCHEMA,
        max_output_tokens=2000,
        temperature=0.2,
        fallback_models=CONFIG["gemini"].get("fallback_chain", []),
    )

    payload = getattr(result, "tool_input", {}) or {}
    if not isinstance(payload, dict):
        payload = {}

    signals_raw = payload.get("signals", [])
    if not isinstance(signals_raw, list):
        signals_raw = []

    # Defensive: strip signals with missing required fields
    cleaned_signals = []
    for s in signals_raw:
        if not isinstance(s, dict):
            continue
        if s.get("category") in ("stylometric", "content", "structural",
                                  "physics_specific") \
                and s.get("severity") in ("weak", "moderate", "strong"):
            cleaned_signals.append(s)

    n_strong = sum(1 for s in cleaned_signals if s.get("severity") == "strong")
    n_moderate = sum(1 for s in cleaned_signals if s.get("severity") == "moderate")
    n_weak = sum(1 for s in cleaned_signals if s.get("severity") == "weak")

    verdict = payload.get("verdict", "unclear")
    if verdict not in ("likely_human", "unclear", "possibly_llm", "likely_llm"):
        verdict = "unclear"

    return LlmDetectionReport(
        doi_label=doi_label,
        verdict=verdict,
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        signals=cleaned_signals,
        reasoning=payload.get("reasoning", ""),
        model=getattr(result, "model", MODEL_NAME),
        n_signals=len(cleaned_signals),
        n_strong=n_strong,
        n_moderate=n_moderate,
        n_weak=n_weak,
        input_tokens=getattr(result, "input_tokens", 0),
        output_tokens=getattr(result, "output_tokens", 0),
    )


def save_report(report: LlmDetectionReport, out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "llm_detection_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{report.doi_label}.json"
    out.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    return out
