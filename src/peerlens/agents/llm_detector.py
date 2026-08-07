"""LLM authorship detector: assess whether a manuscript shows signs of LLM writing.

Single Gemini call with the PDF, prose output, 200-word target, plus a
verdict line. Aligned with the other prose agents (summary / novelty /
structure / reproducibility / domain expert).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient

try:
    from google.genai import types  # type: ignore
except Exception:
    types = None  # type: ignore


VALID_VERDICTS = ("likely_human", "unclear", "possibly_llm", "likely_llm")


@dataclass
class LLMDetectionReport:
    doi_label: str
    verdict: str = "unclear"
    confidence: float = 0.0
    narrative: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


DETECTOR_PROMPT = """You are an editorial assistant assessing whether a scientific manuscript shows evidence of being written by a large language model (LLM), fully or in part. You are given the manuscript as a PDF.

Read the manuscript and look for the following signals:

- Stylistic uniformity: sentences of similar length, similar rhythm, similar transition phrasing across sections
- Hedging boilerplate: overuse of phrases like "it is important to note that", "furthermore, it is worth mentioning", "in summary" in inappropriate places
- Vague or generic scientific claims that lack specificity (numbers, cited authors, mechanism detail)
- Absence of the small idiosyncrasies of human writing: no unusual word choices, no field-specific jargon used in tightly technical ways, no visible authorial voice
- Reference list that reads like a bibliography of vaguely-related papers rather than a targeted citation set
- Factual errors in domain-specific claims that a human expert would not make (fabricated citations, wrong equations, wrong physical constants)
- On the other hand: strong signals of human authorship include equation-level technical detail, non-standard phrasing, specific historical claims, expert-only shortcuts

Write a prose assessment addressed to the editor, in flowing paragraphs, under 200 words. Do not use bullet points or numbered lists. Do not begin your prose with a salutation such as "Dear Editor" or "To the Editor:"; write the commentary directly.

At the end, on a separate line, write exactly one of these verdict formats:

LLM verdict: likely_human
LLM verdict: unclear
LLM verdict: possibly_llm
LLM verdict: likely_llm

On the next line, write your confidence as a decimal between 0.0 and 1.0:

Confidence: 0.85

Use LaTeX notation for any mathematical symbols in your prose. Every mathematical symbol must be wrapped in `$...$`."""


def analyze_llm_authorship(
    doi: str,
    client: Optional[GeminiClient] = None,
    uploaded_file: Any = None,
) -> LLMDetectionReport:
    if client is None:
        client = GeminiClient()

    report = LLMDetectionReport(doi_label=doi)

    if types is None:
        return report

    pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    pdf_bytes = None
    if uploaded_file is None:
        pdf_bytes = pdf_path.read_bytes() if pdf_path.exists() else None
        if pdf_bytes is None:
            return report

    model_name = CONFIG["gemini"]["model_specialist"]
    text, in_tok, out_tok = client.plain_text(
        model=model_name,
        system=DETECTOR_PROMPT,
        user="Please write the LLM-authorship assessment as instructed.",
        temperature=0.2,
        max_output_tokens=16000,
        pdf_bytes=pdf_bytes,
        uploaded_file=uploaded_file,
    )

    text = (text or "").strip()

    verdict = "unclear"
    m = re.search(r"LLM verdict:\s*(\w+)", text, flags=re.IGNORECASE)
    if m:
        v = m.group(1).strip().lower()
        if v in VALID_VERDICTS:
            verdict = v
        text = text[: m.start()].rstrip() + text[m.end():].lstrip("\n")

    confidence = 0.0
    m = re.search(r"Confidence:\s*([0-9]+\.?[0-9]*)", text, flags=re.IGNORECASE)
    if m:
        try:
            confidence = float(m.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            pass
        text = text[: m.start()].rstrip() + text[m.end():].lstrip("\n")

    report.narrative = text.strip()
    report.verdict = verdict
    report.confidence = confidence
    report.model = model_name
    report.input_tokens = in_tok
    report.output_tokens = out_tok
    return report


def save_report(report: LLMDetectionReport, out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "llm_detection_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
