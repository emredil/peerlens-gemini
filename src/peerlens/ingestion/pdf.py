"""PDF parsing with PyMuPDF (fitz)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

import fitz

from peerlens.settings import get_path


@dataclass
class Section:
    level: int
    number: str
    title: str
    text: str


@dataclass
class ParsedPaper:
    doi_label: str
    source_pdf: str
    n_pages: int
    title: str
    abstract: str
    sections: list[Section] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    figure_captions: list[str] = field(default_factory=list)
    table_captions: list[str] = field(default_factory=list)
    equation_count: int = 0
    full_text: str = ""


SECTION_HEADING_RE = re.compile(
    r"^\s*(\d+(?:\.\d+){0,3})\s+([A-Z][A-Za-z0-9 ,\-:'()/]{2,120})\s*$",
    re.MULTILINE,
)

BODY_START_HEADINGS = {
    "Introduction", "Introduction and summary", "Introduction and motivation",
    "Motivation", "Overview", "Preliminaries", "Background",
}

# Minimum characters between a real section heading and the next.
# ToC entries have gaps of ~3-50; real sections have hundreds to thousands.
MIN_REAL_SECTION_GAP = 300


def _extract_full_text(doc: fitz.Document) -> str:
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)


def _guess_title(doc: fitz.Document) -> str:
    if len(doc) == 0:
        return ""
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    candidates = []
    for b in blocks:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = span.get("size", 0)
                if len(text) > 8 and size > 12:
                    candidates.append((size, span["bbox"][1], text))
    if not candidates:
        return ""
    candidates.sort(key=lambda c: (-c[0], c[1]))
    top_size = candidates[0][0]
    title_spans = [(y, t) for (s, y, t) in candidates if abs(s - top_size) < 0.5]
    title_spans.sort(key=lambda x: x[0])
    title = " ".join(t for _, t in title_spans[:6])
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _extract_abstract(full_text: str) -> str:
    m = re.search(
        r"\bAbstract\b\s*[\.\-:]?\s*\n?(.+?)(?=\n\s*(?:Contents|\d+\.?\s+Introduction|\bIntroduction\b|\bKeywords\b|\bCopyright\b))",
        full_text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        text = m.group(1).strip()
        text = re.sub(r"\s+", " ", text)
        return text[:3000]
    return ""


def _extract_sections(full_text: str) -> list[Section]:
    """Extract real section body text, skipping ToC and running headers.

    A heading is 'real' iff its gap to the next heading is >= MIN_REAL_SECTION_GAP.
    We then start from the first heading that names a body anchor (Introduction, etc.),
    stop at References/Acknowledgements/Bibliography, and deduplicate by (number, title).
    """
    all_matches = list(SECTION_HEADING_RE.finditer(full_text))
    if not all_matches:
        return []

    # Keep only matches with substantial gap to the next match.
    real = []
    for i, m in enumerate(all_matches):
        next_start = all_matches[i + 1].start() if i + 1 < len(all_matches) else len(full_text)
        gap = next_start - m.end()
        if gap >= MIN_REAL_SECTION_GAP:
            real.append((m, next_start))

    if not real:
        return []

    # Find the first real body-start heading and drop anything before it.
    body_start_idx = 0
    for i, (m, _) in enumerate(real):
        title = m.group(2).strip()
        if any(title.lower().startswith(a.lower()) for a in BODY_START_HEADINGS):
            body_start_idx = i
            break
    real = real[body_start_idx:]

    raw_sections = []
    for m, next_start in real:
        number = m.group(1)
        title = m.group(2).strip()
        if any(k in title for k in ("References", "Acknowledg", "Bibliography")):
            break
        level = number.count(".") + 1
        body = full_text[m.end():next_start]
        for stop_word in ("\nReferences\n", "\nAcknowledgements\n",
                          "\nAcknowledgments\n", "\nBibliography\n"):
            idx = body.find(stop_word)
            if idx != -1:
                body = body[:idx]
                break
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            raw_sections.append(Section(level=level, number=number, title=title, text=body[:15000]))

    dedup: dict[tuple[str, str], Section] = {}
    for s in raw_sections:
        key = (s.number, s.title)
        if key not in dedup or len(s.text) > len(dedup[key].text):
            dedup[key] = s

    def sort_key(s: Section) -> tuple:
        parts = s.number.split(".")
        return tuple(int(p) if p.isdigit() else 0 for p in parts)

    return sorted(dedup.values(), key=sort_key)


def _extract_references(full_text: str) -> list[str]:
    m = re.search(r"\n\s*References\s*\n", full_text)
    if not m:
        return []
    tail = full_text[m.end():]
    numbered = re.split(r"\n\s*\[\s*\d+\s*\]\s*", tail)
    if len(numbered) > 3:
        refs = [re.sub(r"\s+", " ", r).strip() for r in numbered[1:]]
        refs = [r for r in refs if len(r) > 20][:400]
        return refs
    lines = [ln.strip() for ln in tail.split("\n") if len(ln.strip()) > 20]
    return lines[:400]


def _extract_captions(full_text: str, kind: str) -> list[str]:
    pattern = rf"\b{kind}\s+\d+\s*[:\.]\s*(.+?)(?=\n\n|\n\s*(?:Figure|Table)\s+\d+|$)"
    matches = re.findall(pattern, full_text, re.DOTALL | re.IGNORECASE)
    return [re.sub(r"\s+", " ", m).strip()[:800] for m in matches]


def _count_equations(full_text: str) -> int:
    return len(re.findall(r"\((\d+(?:\.\d+){0,2})\)", full_text))


def parse_pdf(pdf_path: Path, doi_label: str) -> ParsedPaper:
    doc = fitz.open(pdf_path)
    try:
        full_text = _extract_full_text(doc)
        paper = ParsedPaper(
            doi_label=doi_label,
            source_pdf=str(pdf_path),
            n_pages=len(doc),
            title=_guess_title(doc),
            abstract=_extract_abstract(full_text),
            sections=_extract_sections(full_text),
            references=_extract_references(full_text),
            figure_captions=_extract_captions(full_text, "Figure"),
            table_captions=_extract_captions(full_text, "Table"),
            equation_count=_count_equations(full_text),
            full_text=full_text,
        )
    finally:
        doc.close()
    return paper


def save_parsed(paper: ParsedPaper) -> Path:
    out_dir = get_path("processed_data") / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{paper.doi_label}.json"
    payload = asdict(paper)
    payload["full_text"] = payload["full_text"][:200_000]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def parse_and_save(pdf_path: Path, doi_label: str) -> ParsedPaper:
    paper = parse_pdf(pdf_path, doi_label)
    save_parsed(paper)
    return paper
