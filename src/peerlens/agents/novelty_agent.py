"""Novelty agent: assess a manuscript's novelty against the current literature.

Uses Gemini with Google Search grounding to retrieve and reason about related
work in a single call. The old Semantic Scholar + local corpus retrieval
pipeline (which was Physics-only and mis-routed identifiers for non-SciPost
papers) has been replaced by domain-agnostic web grounding.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any, Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient

try:
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover - import guarded so tests can stub
    types = None  # type: ignore


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NoveltyFinding:
    """Minimal placeholder; kept for aggregator compatibility.

    The new grounded workflow no longer produces structured findings; the
    novelty assessment is a single prose narrative. The list is retained
    (always empty) so that the aggregator\'s `_novelty_verdict` helper
    still returns "clean" instead of crashing on a missing field.
    """
    kind: str = ""
    severity: str = ""
    location: str = ""
    evidence: str = ""


@dataclass
class NoveltyReport:
    doi_label: str
    narrative: str = ""
    verdict: str = ""  # free-text label from the prose (e.g. "clearly novel")
    reference_year_median: Optional[int] = None
    reference_year_max: Optional[int] = None
    n_findings: int = 0
    findings: list[NoveltyFinding] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Reference-year statistics from the manuscript\'s own bibliography
# ---------------------------------------------------------------------------

def _extract_year(ref: str) -> Optional[int]:
    # Strip arXiv identifiers (e.g. "arXiv:2093.05231" or "1903.03686")
    # before scanning for a year, because arXiv IDs share the same
    # 4-digit prefix pattern as years and would otherwise be treated
    # as reference years.
    cleaned = re.sub(r"arXiv[:\s]*\d{4}\.\d{4,5}", " ", ref, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d{4}\.\d{4,5}\b", " ", cleaned)
    m = re.search(r"\b(19|20)\d{2}\b(?!\.\d)", cleaned)
    return int(m.group(0)) if m else None


def reference_year_stats(paper_refs: list[str]) -> tuple[Optional[int], Optional[int]]:
    years = [_extract_year(r) for r in paper_refs]
    upper = date.today().year + 1
    years = [y for y in years if y is not None and 1900 <= y <= upper]
    if not years:
        return None, None
    years.sort()
    median = years[len(years) // 2]
    return median, max(years)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

NOVELTY_REVIEWER_PROMPT = """You are reviewing a scientific manuscript for a journal editor. Your task is to assess its novelty relative to the existing literature.

You are given the manuscript itself (attached as PDF). Use the Google Search tool available to you to find related prior work in the same subfield. Search with keywords drawn from the manuscript\'s title, abstract, and main claims. Consider both preprints (arXiv, ChemRxiv, bioRxiv) and published journal articles.

Write a novelty assessment as a working referee would write it, in prose, addressed to the editor. Cover the following in a natural flow:

1. What is the manuscript\'s central claim of novelty? Read the abstract and introduction carefully; identify what the authors say is new.

2. How does this claim hold up against the related prior work you found via search? Briefly compare against the two or three closest hits: what does the current manuscript do that this related work does not, or what does the related work already do that the current manuscript re-derives without acknowledgement.

3. Is the current manuscript missing citations to any of the closely related papers you found? Check the reference list of the manuscript against the top hits from your search. If a highly relevant paper is not cited, flag it briefly.

4. Overall judgment: is the manuscript\'s novelty claim substantiated? At the end of your prose, on a separate line, write your overall judgment exactly in this format (no extra words):

Novelty verdict: clearly novel
Novelty verdict: incremental
Novelty verdict: overlaps with prior work

Keep the assessment under 200 words. Do not use bullet points or numbered lists in your output. Do not begin your prose with a salutation such as "Dear Editor" or "To the Editor:"; write the commentary directly.

Use LaTeX notation for any mathematical symbols and equations in your prose. Write `$\\varepsilon$` not `\u03b5`, `$\\mu$` not `\u00b5`, `$\\pi$` not `\u03c0`. Every mathematical symbol must be wrapped in `$...$` so the report renders cleanly in markdown with math support."""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_novelty(
    doi: str,
    client: Optional[GeminiClient] = None,
    uploaded_file: Any = None,
) -> NoveltyReport:
    if client is None:
        client = GeminiClient()

    report = NoveltyReport(doi_label=doi)

    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not parsed_path.exists():
        return report
    parsed = json.loads(parsed_path.read_text())

    # Reference-year statistics from the manuscript\'s own bibliography
    refs = parsed.get("references", []) or []
    med_yr, max_yr = reference_year_stats(refs)
    report.reference_year_median = med_yr
    report.reference_year_max = max_yr

    if types is None:
        # google-genai not importable; leave narrative empty.
        return report

    # PDF: prefer cached upload, fall back to inline bytes.
    pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    pdf_bytes = None
    if uploaded_file is None:
        pdf_bytes = pdf_path.read_bytes() if pdf_path.exists() else None

    parts: list = []
    if uploaded_file is not None:
        file_uri = getattr(uploaded_file, "uri", None) or getattr(uploaded_file, "name", None)
        if file_uri is None:
            raise ValueError("uploaded_file has no uri or name attribute")
        parts.append(types.Part.from_uri(file_uri=file_uri, mime_type="application/pdf"))
    elif pdf_bytes is not None:
        parts.append(types.Part.from_bytes(mime_type="application/pdf", data=pdf_bytes))
    parts.append(types.Part(text="Please write the novelty assessment as instructed."))

    config = types.GenerateContentConfig(
        system_instruction=NOVELTY_REVIEWER_PROMPT,
        temperature=0.3,
        max_output_tokens=8000,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )

    model_name = CONFIG["gemini"]["model_specialist"]
    resp = client._client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=parts)],
        config=config,
    )

    text = ""
    try:
        text = resp.text or ""
    except Exception:
        for c in getattr(resp, "candidates", []) or []:
            for prt in getattr(c.content, "parts", []) or []:
                if hasattr(prt, "text"):
                    text += prt.text or ""

    text = text.strip()

    # Extract "Novelty verdict: ..." tail (if present) and strip it from narrative.
    verdict = ""
    m = re.search(r"^Novelty verdict:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        verdict = m.group(1).strip()
        text = text[: m.start()].rstrip()

    report.narrative = text
    report.verdict = verdict

    usage = getattr(resp, "usage_metadata", None)
    report.model = model_name
    report.input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    report.output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

    return report


def save_report(report: NoveltyReport, out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "novelty_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
