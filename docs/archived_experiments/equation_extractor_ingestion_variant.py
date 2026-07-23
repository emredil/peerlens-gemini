"""Extract numbered equations from a PDF using Gemini Flash vision.

Strategy:
- Render only pages that contain numbered-equation markers (from PyMuPDF text).
- Send each such page to Gemini Flash as PNG + a strict JSON prompt.
- Persist per-paper JSON with (equation_label, latex, context_hint, page).

This module is separate from the Claude-based pipeline; Gemini here only does
OCR-style extraction. Reasoning about the extracted equations happens later,
in the Claude-based math checker.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import fitz
from google import genai
from google.genai import types

from peerlens.settings import CONFIG, get_path


PAGE_EQUATION_MARKER = re.compile(r"\((\d+(?:\.\d+){0,2})\)")


@dataclass
class ExtractedEquation:
    equation_label: str
    latex: str
    context_hint: str
    page: int


@dataclass
class ExtractedPaperEquations:
    doi_label: str
    n_pages_scanned: int
    n_pages_with_equations: int
    equations: list[ExtractedEquation] = field(default_factory=list)
    model: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0


PROMPT = """You are looking at a page from a physics paper. Extract every
numbered displayed equation you see on this page.

For each equation, return:
- equation_label: the parenthesized number, e.g. "(3)" or "(2.7)"
- latex: the equation body as valid LaTeX (do NOT include the label)
- context_hint: one short phrase describing what the equation represents,
  based on the surrounding text

Ignore inline math. Only capture DISPLAYED numbered equations.

Return strictly JSON in the shape:
{"equations": [
  {"equation_label": "(3)", "latex": "H = -J \\\\sum ...", "context_hint": "..."}
]}
If the page has no numbered equations, return {"equations": []}."""


def _pages_with_equations(doc: fitz.Document) -> list[int]:
    """Return 0-indexed page numbers that likely contain numbered equations.

    Heuristic: a page containing at least 2 '(N)' markers where N is a small
    integer that looks like an equation label.
    """
    out = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        matches = PAGE_EQUATION_MARKER.findall(text)
        # filter to labels that plausibly are equation numbers (1-999, or 1.1, 2.7 etc.)
        labels = [m for m in matches if _plausible_equation_label(m)]
        if len(labels) >= 2:
            out.append(i)
    return out


def _plausible_equation_label(m: str) -> bool:
    """A label like '3' or '2.7' or 'A.1'."""
    parts = m.split(".")
    if not all(p.isdigit() for p in parts):
        return False
    try:
        head = int(parts[0])
        return 1 <= head <= 200
    except ValueError:
        return False


def _render_page(page: fitz.Page, dpi: int = 200) -> bytes:
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


from google.genai.errors import ServerError, ClientError

# Fallback model chain: if the -latest alias is under load, try a pinned model.
FALLBACK_MODELS: list[str] = []

# Backoff schedule for transient Gemini errors (seconds).
RETRY_DELAYS = [5, 15]  # shorter total wait so failed pages fail fast


def _try_generate(
    client: genai.Client,
    image_bytes: bytes,
    model_name: str,
) -> tuple[str | None, int, int]:
    """Single API call. Returns (response_text_or_None, in_tok, out_tok).

    Returns None text on transient failure so the caller can retry or fallback.
    Raises on non-transient errors (auth, invalid request).
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=CONFIG.get("gemini", {}).get("temperature", 0.0),
            ),
        )
    except ServerError as e:
        # 5xx errors: treat as transient
        return None, 0, 0
    except ClientError as e:
        # 429 rate-limit is transient; other 4xx are real problems
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return None, 0, 0
        raise

    in_tok = 0
    out_tok = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
        in_tok = response.usage_metadata.prompt_token_count or 0
        out_tok = response.usage_metadata.candidates_token_count or 0
    return response.text, in_tok, out_tok


def _extract_from_page(
    client: genai.Client,
    image_bytes: bytes,
    model_name: str,
    page_num_1based: int,
    console_log=None,
) -> tuple[list[ExtractedEquation], int, int, bool]:
    """Call Gemini on one page image with retry + model fallback.

    Returns (equations, in_tok, out_tok, success_flag).
    success_flag=False means the page was skipped after all retries.
    """
    models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]

    for mi, model in enumerate(models_to_try):
        for attempt, delay in enumerate([0] + RETRY_DELAYS):
            if delay:
                if console_log:
                    console_log(f"    retry {attempt}/{len(RETRY_DELAYS)} on {model} after {delay}s")
                time.sleep(delay)
            text, in_tok, out_tok = _try_generate(client, image_bytes, model)
            if text is not None:
                # success
                try:
                    payload = json.loads(text)
                    eqs = payload.get("equations", [])
                except (json.JSONDecodeError, TypeError):
                    eqs = []

                result = []
                for e in eqs:
                    result.append(ExtractedEquation(
                        equation_label=str(e.get("equation_label", "")),
                        latex=str(e.get("latex", "")),
                        context_hint=str(e.get("context_hint", "")),
                        page=page_num_1based,
                    ))
                return result, in_tok, out_tok, True
        # exhausted retries on this model; try next fallback model
        if console_log and mi + 1 < len(models_to_try):
            console_log(f"    falling back to {models_to_try[mi+1]}")

    # all models exhausted
    if console_log:
        console_log(f"    page {page_num_1based} skipped (all models unavailable)")
    return [], 0, 0, False


def extract_equations_from_pdf(
    pdf_path: Path,
    doi_label: str,
    client: genai.Client | None = None,
    max_pages: int | None = None,
    per_page_sleep: float = 1.5,
) -> ExtractedPaperEquations:
    """Extract equations from a whole PDF. Only pages likely to contain equations are scanned."""
    if client is None:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    model_name = CONFIG["gemini"]["model_equation_extractor"]

    doc = fitz.open(pdf_path)
    try:
        candidate_pages = _pages_with_equations(doc)
        if max_pages is not None:
            candidate_pages = candidate_pages[:max_pages]

        all_eqs: list[ExtractedEquation] = []
        total_in = 0
        total_out = 0
        pages_skipped = 0
        for i in candidate_pages:
            image_bytes = _render_page(doc[i])
            eqs, in_tok, out_tok, ok = _extract_from_page(
                client, image_bytes, model_name, i + 1
            )
            if not ok:
                pages_skipped += 1
            all_eqs.extend(eqs)
            total_in += in_tok
            total_out += out_tok
            if per_page_sleep:
                time.sleep(per_page_sleep)

        return ExtractedPaperEquations(
            doi_label=doi_label,
            n_pages_scanned=len(candidate_pages),
            n_pages_with_equations=sum(1 for i in candidate_pages),
            equations=all_eqs,
            model=model_name,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )
    finally:
        doc.close()


def save_equations(result: ExtractedPaperEquations, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "equations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.doi_label}.json"
    payload = asdict(result)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path
