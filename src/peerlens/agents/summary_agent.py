"""Summary Agent: writes a plain-language paper summary for the editor.

One Gemini call, PDF cache-backed, no verdict, just prose. Feeds the
Summary section of the referee report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


@dataclass
class SummaryReport:
    doi_label: str
    narrative: str = ""
    n_paragraphs: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


SUMMARY_PROMPT = """You are writing a plain summary of a scientific manuscript for a journal editor. The editor needs to know, in two or three short paragraphs, what this paper is about, so that the editor can make a first-pass decision about whether to send it out for detailed review.

You are given:
- The manuscript itself (attached as PDF)
- The manuscript is approximately <<n_pages>> pages long
- The paper's field: <<field>>
- The paper's subdomain: <<subdomain>>

Write <<n_para>> short paragraph(s) covering:

1. What problem does the paper address, and why does it matter within its subdomain?
2. What is the paper's main contribution or result?
3. What method or approach does the paper use?

Write in flowing prose, addressed to the editor. No bullet points, no numbered lists, no marker lines. Do not evaluate the paper's quality; that is done elsewhere. Just describe what the paper does.

Use LaTeX notation for any mathematical symbols and equations. Write `$\\varepsilon$` not `ε`, `$\\mu$` not `µ`, `$g_{tt}$` not `g_tt`. Every mathematical symbol in the summary must be wrapped in `$...$` so the summary renders cleanly in markdown with math support."""


def analyze_summary(
    doi: str,
    client: Optional[GeminiClient] = None,
    uploaded_file: Any = None,
) -> SummaryReport:
    if client is None:
        client = GeminiClient()

    report = SummaryReport(doi_label=doi)

    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not parsed_path.exists():
        return report
    parsed = json.loads(parsed_path.read_text())

    # Load classification for field/subdomain context.
    cls_path = get_path("processed_data") / "classifications" / f"{doi}.json"
    field = "physics"
    subdomain = "unknown"
    if cls_path.exists():
        cls_data = json.loads(cls_path.read_text())
        field = cls_data.get("field_top", "physics")
        subs = cls_data.get("subdomains", [])
        if subs:
            subdomain = subs[0]

    # Paragraph budget by paper length.
    n_pages = parsed.get("n_pages", 0) or 10
    if n_pages < 10:
        n_para = 2
    elif n_pages < 25:
        n_para = 2
    else:
        n_para = 3
    report.n_paragraphs = n_para

    system_prompt = (
        SUMMARY_PROMPT
        .replace("<<n_pages>>", str(n_pages))
        .replace("<<field>>", field)
        .replace("<<subdomain>>", subdomain)
        .replace("<<n_para>>", str(n_para))
    )

    # PDF handling: prefer uploaded_file (cached), fall back to inline bytes.
    pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    pdf_bytes = None
    if uploaded_file is None:
        pdf_bytes = pdf_path.read_bytes() if pdf_path.exists() else None

    user_prose = (
        f"Please write the {n_para}-paragraph summary of this "
        f"{field} manuscript ({subdomain}, {n_pages} pages) as instructed."
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

    report.narrative = (text or "").strip()
    report.model = model_name
    report.input_tokens = in_tok
    report.output_tokens = out_tok
    return report


def save_report(report: SummaryReport, out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "summary_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
