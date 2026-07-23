"""Semantic Scholar API client for prior-art / related-work checks.

Uses the free public API (no key needed, limited to a modest rate).
Docs: https://api.semanticscholar.org/api-docs/graph
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx


BASE_URL = "https://api.semanticscholar.org/graph/v1"
USER_AGENT = "PeerLens/0.0.1 (research prototype)"
RATE_LIMIT_SLEEP = 1.0  # public API is throttled to a few req/sec


@dataclass
class ScholarPaper:
    paper_id: str
    title: str
    year: Optional[int]
    venue: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    citation_count: int = 0
    reference_count: int = 0
    external_ids: dict = field(default_factory=dict)


def _headers():
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def search_by_query(
    query: str,
    limit: int = 10,
    year_from: Optional[int] = None,
    field_of_study: Optional[str] = "Physics",
) -> list[ScholarPaper]:
    """Full-text search — good for finding related work by title/abstract keywords."""
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,year,venue,abstract,authors,citationCount,referenceCount,externalIds",
    }
    if year_from:
        params["year"] = f"{year_from}-"
    if field_of_study:
        params["fieldsOfStudy"] = field_of_study

    try:
        with httpx.Client(headers=_headers(), timeout=30.0) as client:
            resp = client.get(f"{BASE_URL}/paper/search", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError:
        return []

    results = []
    for item in data.get("data", []):
        results.append(_paper_from_json(item))
    time.sleep(RATE_LIMIT_SLEEP)
    return results


def get_paper(paper_id: str) -> Optional[ScholarPaper]:
    """Fetch a single paper by ID (Semantic Scholar ID or DOI:...)"""
    params = {
        "fields": "paperId,title,year,venue,abstract,authors,citationCount,referenceCount,externalIds",
    }
    try:
        with httpx.Client(headers=_headers(), timeout=30.0) as client:
            resp = client.get(f"{BASE_URL}/paper/{paper_id}", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError:
        return None
    time.sleep(RATE_LIMIT_SLEEP)
    return _paper_from_json(data)


def get_references(paper_id: str, limit: int = 100) -> list[ScholarPaper]:
    """Get papers this one references."""
    params = {
        "fields": "paperId,title,year,venue,authors,citationCount,externalIds",
        "limit": limit,
    }
    try:
        with httpx.Client(headers=_headers(), timeout=30.0) as client:
            resp = client.get(f"{BASE_URL}/paper/{paper_id}/references", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError:
        return []

    out = []
    for r in data.get("data", []):
        p = r.get("citedPaper") or {}
        if p:
            out.append(_paper_from_json(p))
    time.sleep(RATE_LIMIT_SLEEP)
    return out


def get_recent_related(paper_id: str, limit: int = 20, year_from: int = 2023) -> list[ScholarPaper]:
    """Papers similar to this one, from the specified year onward.

    Uses Semantic Scholar's 'recommendations' endpoint which surfaces
    papers similar in topic/citation graph.
    """
    try:
        with httpx.Client(headers=_headers(), timeout=30.0) as client:
            resp = client.get(
                f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}",
                params={"limit": limit, "fields": "paperId,title,year,venue,authors,citationCount,externalIds"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError:
        return []
    time.sleep(RATE_LIMIT_SLEEP)

    out = []
    for p in data.get("recommendedPapers", []):
        year = p.get("year")
        if year is not None and year < year_from:
            continue
        out.append(_paper_from_json(p))
    return out


def _paper_from_json(item: dict) -> ScholarPaper:
    return ScholarPaper(
        paper_id=item.get("paperId", ""),
        title=item.get("title", "") or "",
        year=item.get("year"),
        venue=item.get("venue", "") or "",
        abstract=item.get("abstract", "") or "",
        authors=[a.get("name", "") for a in item.get("authors", []) or []],
        citation_count=item.get("citationCount", 0) or 0,
        reference_count=item.get("referenceCount", 0) or 0,
        external_ids=item.get("externalIds", {}) or {},
    )
