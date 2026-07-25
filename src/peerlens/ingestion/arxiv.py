"""arXiv fetcher: given an arXiv identifier or URL, download the PDF and
pull basic metadata (title, authors, abstract, categories).

Uses the public arXiv API (https://info.arxiv.org/help/api/index.html) —
no key required. Convention is 1 request every 3 seconds; we serialize
one paper at a time so the built-in delay is enough.
"""
from __future__ import annotations

import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class ArxivMetadata:
    arxiv_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    published: str = ""
    updated: str = ""
    doi: str = ""
    primary_category: str = ""


def normalize_arxiv_id(raw: str) -> str:
    """Accept a raw string and return the canonical arXiv id.

    Accepts formats such as:
      - 2401.12345
      - 2401.12345v2
      - arXiv:2401.12345
      - https://arxiv.org/abs/2401.12345
      - https://arxiv.org/pdf/2401.12345
      - hep-th/0512345
    """
    s = raw.strip()
    s = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", s)
    s = re.sub(r"^arxiv:", "", s, flags=re.IGNORECASE)
    s = s.rstrip("/")
    if s.lower().endswith(".pdf"):
        s = s[:-4]
    return s


def fetch_metadata(arxiv_id: str, retries: int = 3) -> Optional[ArxivMetadata]:
    """Query the arXiv API for one paper."""
    aid = normalize_arxiv_id(arxiv_id)
    url = f"{ARXIV_API}?id_list={aid}&max_results=1"
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = resp.read()
            break
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"arXiv API failed for {aid}: {last_err}")

    root = ET.fromstring(data)
    entry = root.find("atom:entry", _NS)
    if entry is None:
        return None

    def _text(elem, path):
        e = elem.find(path, _NS)
        return (e.text or "").strip() if e is not None else ""

    meta = ArxivMetadata(arxiv_id=aid)
    meta.title = re.sub(r"\s+", " ", _text(entry, "atom:title"))
    meta.abstract = re.sub(r"\s+", " ", _text(entry, "atom:summary"))
    meta.published = _text(entry, "atom:published")
    meta.updated = _text(entry, "atom:updated")

    for author in entry.findall("atom:author", _NS):
        name = _text(author, "atom:name")
        if name:
            meta.authors.append(name)

    for cat in entry.findall("atom:category", _NS):
        term = cat.get("term")
        if term:
            meta.categories.append(term)

    pc = entry.find("arxiv:primary_category", _NS)
    if pc is not None:
        meta.primary_category = pc.get("term", "")

    doi_elem = entry.find("arxiv:doi", _NS)
    if doi_elem is not None and doi_elem.text:
        meta.doi = doi_elem.text.strip()

    return meta


def download_pdf(arxiv_id: str, dest: Path, retries: int = 3) -> Path:
    """Download the PDF for the given arXiv id to `dest`."""
    aid = normalize_arxiv_id(arxiv_id)
    url = ARXIV_PDF.format(arxiv_id=aid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "PeerLens/0.1 (research prototype)",
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                dest.write_bytes(resp.read())
            return dest
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"arXiv PDF download failed for {aid}: {last_err}")


def arxiv_id_to_doi_label(arxiv_id: str) -> str:
    """Canonical filename stem for this paper in our data tree.

    Examples:
      2401.12345      -> arxiv_2401.12345
      hep-th/0512345  -> arxiv_hep-th_0512345
    """
    aid = normalize_arxiv_id(arxiv_id)
    safe = aid.replace("/", "_")
    return f"arxiv_{safe}"
