"""eLife Reviewed Preprints ingestion client.

Fetches paper metadata (title, authors, subjects, eLife assessment) from
the public eLife API, and downloads the underlying preprint PDF from
bioRxiv / medRxiv using the preprint DOI.

The eLife Assessment is our ground truth for biology testing: it contains
a prose editorial evaluation plus two standardised tags:
  significance: "important" / "valuable" / "useful" / "landmark" / "fundamental"
  strength:     "convincing" / "solid" / "compelling" / "incomplete" /
                "exceptional" / "inadequate"

Public API base: https://api.elifesciences.org
Docs (RAML):     https://github.com/elifesciences/api-raml
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

import httpx

from peerlens.settings import get_path


ELIFE_API_BASE = "https://api.elifesciences.org"
ELIFE_ACCEPT_LIST = "application/vnd.elife.reviewed-preprint-list+json; version=1"
ELIFE_ACCEPT_ITEM = "application/vnd.elife.reviewed-preprint+json; version=1"
USER_AGENT = "PeerLens/0.0.1 (research prototype; peer-review evaluation)"
RATE_LIMIT_SLEEP = 1.0


@dataclass
class ElifePaper:
    """Slim metadata record for one reviewed preprint."""
    msid: str                    # eLife manuscript ID (e.g. "111930")
    doi: str                     # bioRxiv/medRxiv preprint DOI
    version: int
    title: str
    author_line: str
    published: str
    subjects: list[str] = field(default_factory=list)
    assessment_text: str = ""    # prose eLife Assessment
    significance: list[str] = field(default_factory=list)
    strength: list[str] = field(default_factory=list)

    @property
    def doi_label(self) -> str:
        """Filesystem-safe label used across the pipeline."""
        return f"elife_{self.msid}"


def _headers(accept: str) -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": accept}


def _paper_from_item(item: dict) -> ElifePaper:
    """Extract an ElifePaper from one item in the /reviewed-preprints response."""
    assessment = item.get("elifeAssessment") or {}
    content = assessment.get("content") or []
    prose = ""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "paragraph":
            prose += (block.get("text") or "") + "\n\n"
    return ElifePaper(
        msid=str(item.get("id", "")),
        doi=item.get("doi", "") or "",
        version=int(item.get("version") or 1),
        title=item.get("title", "") or "",
        author_line=item.get("authorLine", "") or "",
        published=item.get("published", "") or "",
        subjects=[s.get("name", "") for s in (item.get("subjects") or [])],
        assessment_text=prose.strip(),
        significance=list(assessment.get("significance") or []),
        strength=list(assessment.get("strength") or []),
    )


def list_reviewed_preprints(
    max_records: int = 100,
    per_page: int = 50,
    order: str = "desc",
) -> Iterator[ElifePaper]:
    """Yield ElifePaper records from the eLife API, newest first.

    order: 'desc' (newest first) or 'asc' (oldest first).
    Pagination follows the eLife API's page/per-page convention.
    """
    per_page = min(per_page, 100)
    fetched = 0
    page = 1
    with httpx.Client(headers=_headers(ELIFE_ACCEPT_LIST), timeout=30.0) as client:
        while fetched < max_records:
            resp = client.get(
                f"{ELIFE_API_BASE}/reviewed-preprints",
                params={"page": page, "per-page": per_page, "order": order},
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", []) or []
            if not items:
                return
            for item in items:
                if fetched >= max_records:
                    return
                yield _paper_from_item(item)
                fetched += 1
            page += 1
            time.sleep(RATE_LIMIT_SLEEP)


def _biorxiv_pdf_urls(doi: str) -> list[str]:
    """Return candidate PDF URLs for a bioRxiv / medRxiv DOI.

    bioRxiv DOIs look like "10.1101/2025.09.29.679309". PDF is:
      https://www.biorxiv.org/content/10.1101/2025.09.29.679309v1.full.pdf
    We try v1..v3 in order; return them all so the caller can attempt each.
    """
    if not doi.startswith("10.1101/") and not doi.startswith("10.64898/"):
        # Some newer DOIs use eLife's own 10.64898 prefix; those don't have
        # a bioRxiv fallback and we skip them.
        return []
    urls = []
    for host in ("www.biorxiv.org", "www.medrxiv.org"):
        for v in (1, 2, 3):
            urls.append(f"https://{host}/content/{doi}v{v}.full.pdf")
    return urls


def download_pdf(paper: ElifePaper, force: bool = False) -> Optional[Path]:
    """Download the preprint PDF to data/raw/pdf/elife_<msid>.pdf.

    Returns the path on success, None if all candidate URLs fail.
    """
    out_dir = get_path("raw_data") / "pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{paper.doi_label}.pdf"
    if out_path.exists() and not force:
        return out_path

    urls = _biorxiv_pdf_urls(paper.doi)
    if not urls:
        return None

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True) as client:
        for url in urls:
            try:
                resp = client.get(url)
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
                    out_path.write_bytes(resp.content)
                    time.sleep(RATE_LIMIT_SLEEP)
                    return out_path
            except httpx.HTTPError:
                continue
            time.sleep(0.3)
    return None


def save_metadata(paper: ElifePaper, out_dir: Optional[Path] = None) -> Path:
    """Save the paper metadata + eLife Assessment to disk.

    Written to data/processed/elife_reviews/<doi_label>.json so it lives
    alongside our own referee reports and can serve as ground truth for
    comparison.
    """
    if out_dir is None:
        out_dir = get_path("processed_data") / "elife_reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{paper.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(paper), f, indent=2, ensure_ascii=False)
    return out_path
