"""Equation Extractor: PDF page → numbered LaTeX via Gemini vision (new SDK)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz
from dotenv import load_dotenv
from google import genai
from google.genai import types

from peerlens.settings import get_path

load_dotenv(Path(__file__).resolve().parents[3] / ".env")
_CLIENT: Optional[genai.Client] = None


def _client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


VISION_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]


EXTRACT_PROMPT = """You are an OCR assistant for physics manuscripts.

Look at this page. Find EVERY equation label that appears in parentheses on
the right-hand side of a display equation, like (5), (6), (7), (44), (48),
(3.14). For each such label, extract the full equation it belongs to as
LaTeX and return the list.

Output format is a JSON list. Each item:
  {"number": "7", "latex": "H_M = \\\\sum_m h_m + \\\\sum_m \\\\beta_{2m-3} h_{2m-2} h_{2m-3} h_{2m}"}

CRITICAL RULES:
1. If you see the label (5), (6), (7) ANYWHERE on the page at the end of an
   equation, you MUST include it. Even if the equation is short like
   "H = \\\\sum_m h_m , (6)" — that IS a numbered equation, DO include it.
2. Preserve exact LaTeX. Use \\\\sum, \\\\int, \\\\beta, \\\\alpha, subscripts with
   {}, superscripts with ^{}, overlines with \\\\bar{...} or \\\\overline{...}.
3. Greek and math operators must be LaTeX macros.
4. Multi-line equations (aligned): return the whole thing as one entry.
5. Return ONLY the JSON list. No prose, no code fences.
6. Return [] ONLY if the page has zero numbered display equations."""


@dataclass
class Equation:
    number: str
    latex: str


@dataclass
class PageExtraction:
    doi_label: str
    page_index: int
    equations: list[Equation] = field(default_factory=list)
    model_used: str = ""
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


def _cache_path(doi: str, page_index: int) -> Path:
    out_dir = get_path("processed_data") / "equations"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{doi}_p{page_index:03d}.json"


def _render_page_png(pdf_path: Path, page_index: int, zoom: float = 2.0) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()


def _parse_response_text(text: str) -> list[Equation]:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: t.rfind("```")]
        t = t.strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and "number" in item and "latex" in item:
            out.append(Equation(number=str(item["number"]), latex=str(item["latex"])))
    return out


def extract_page(
    doi_label: str,
    pdf_path: Path,
    page_index: int,
    use_cache: bool = True,
    force_refresh: bool = False,
    timeout_seconds: float = 45.0,
) -> PageExtraction:
    cache = _cache_path(doi_label, page_index)
    if use_cache and cache.exists() and not force_refresh:
        d = json.loads(cache.read_text())
        return PageExtraction(
            doi_label=d["doi_label"],
            page_index=d["page_index"],
            equations=[Equation(**e) for e in d["equations"]],
            model_used=d.get("model_used", "cache"),
            latency_seconds=d.get("latency_seconds", 0.0),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            error=d.get("error", ""),
        )

    png = _render_page_png(pdf_path, page_index)
    client = _client()
    last_err = ""

    for model_name in VISION_MODELS:
        try:
            t0 = time.time()
            resp = client.models.generate_content(
                model=model_name,
                contents=[types.Content(role="user", parts=[
                    types.Part.from_bytes(mime_type="image/png", data=png),
                    types.Part(text=EXTRACT_PROMPT),
                ])],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                ),
            )
            dt = time.time() - t0

            text = ""
            try:
                text = resp.text or ""
            except Exception:
                for c in getattr(resp, "candidates", []) or []:
                    for p in getattr(c.content, "parts", []) or []:
                        if hasattr(p, "text"):
                            text += (p.text or "")

            equations = _parse_response_text(text)
            usage = getattr(resp, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
            out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

            result = PageExtraction(
                doi_label=doi_label,
                page_index=page_index,
                equations=equations,
                model_used=model_name,
                latency_seconds=dt,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            cache.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            return result
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            s = last_err.lower()
            if any(t in s for t in ["quota", "429", "resource exhausted", "unavailable", "503", "500", "404"]):
                time.sleep(0.5)
                continue
            time.sleep(0.2)
            continue

    result = PageExtraction(
        doi_label=doi_label, page_index=page_index,
        error=last_err or "all models failed",
    )
    cache.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return result


def _normalize_eq_number(raw: str) -> str:
    s = raw.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


_DEFINITION_HINT_CHARS = set(",.=+-*/±≡≥≤:")


def _is_definition_context(text: str, match_start: int) -> bool:
    window_start = max(0, match_start - 40)
    ctx = text[window_start:match_start]
    ctx_stripped = ctx.rstrip()
    if not ctx_stripped:
        return False
    lower = ctx.lower()
    for kw in ("eq.", "eq ", "eqs.", "equation", "relation", "condition",
              "identity", "using", "from", "into", "of", "the", "in", "and",
              "see", "with", "via", "by"):
        if lower.rstrip().endswith(kw):
            return False
    last = ctx_stripped[-1]
    if last in _DEFINITION_HINT_CHARS:
        return True
    if last in "})]":
        return True
    return False


def find_pages_for_equations(pdf_path: Path, equation_numbers: list[str]) -> dict[str, list[int]]:
    normalized_wanted = {_normalize_eq_number(eq) for eq in equation_numbers}
    doc = fitz.open(pdf_path)
    definitions: dict[str, list[int]] = {eq: [] for eq in normalized_wanted}
    any_occurrence: dict[str, list[int]] = {eq: [] for eq in normalized_wanted}
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            for eq in normalized_wanted:
                needle = f"({eq})"
                start = 0
                found_on_page = False
                while True:
                    idx = text.find(needle, start)
                    if idx == -1:
                        break
                    if not found_on_page:
                        any_occurrence[eq].append(i)
                        found_on_page = True
                    if _is_definition_context(text, idx):
                        if i not in definitions[eq]:
                            definitions[eq].append(i)
                    start = idx + 1
    finally:
        doc.close()

    result: dict[str, list[int]] = {}
    for eq in normalized_wanted:
        if definitions[eq]:
            result[eq] = definitions[eq]
        elif any_occurrence[eq]:
            result[eq] = any_occurrence[eq]
    return result


def extract_equations_for_claim(
    doi_label: str,
    pdf_path: Path,
    equation_numbers: list[str],
) -> dict[str, Equation]:
    page_map = find_pages_for_equations(pdf_path, equation_numbers)
    all_pages: set[int] = set()
    for pages in page_map.values():
        all_pages.update(pages)
    page_extractions: dict[int, PageExtraction] = {}
    for p in sorted(all_pages):
        page_extractions[p] = extract_page(doi_label, pdf_path, p)

    eq_lookup: dict[str, Equation] = {}
    for eq, candidate_pages in page_map.items():
        for p in candidate_pages:
            for e in page_extractions[p].equations:
                if _normalize_eq_number(e.number) == eq:
                    eq_lookup[eq] = e
                    break
            if eq in eq_lookup:
                break
    return eq_lookup
