"""Corpus embedding index using ChromaDB.

Indexes every parsed SciPost paper in data/processed/parsed/ into a local
Chroma collection, keyed by DOI. Embeddings are computed from title + abstract.

The index is used by the novelty agent to answer:
  "What SciPost papers are semantically closest to this manuscript?"
This complements Semantic Scholar, which sometimes misses very recent papers
or papers whose venue metadata is incomplete.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from peerlens.settings import get_path


COLLECTION_NAME = "scipost_corpus"


@dataclass
class SimilarPaper:
    doi_label: str
    title: str
    similarity: float   # 1 - distance; higher is more similar
    abstract_snippet: str = ""


def _client() -> chromadb.PersistentClient:
    """Chroma persistent client — data lives in data/corpus/chroma/."""
    path = get_path("vector_store") / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection():
    c = _client()
    return c.get_or_create_collection(name=COLLECTION_NAME)


def build_index(force_rebuild: bool = False) -> int:
    """Build (or refresh) the corpus index.

    Returns the number of documents indexed.
    """
    coll = _collection()

    parsed_dir = get_path("processed_data") / "parsed"
    parsed_files = sorted(parsed_dir.glob("SciPostPhys.*.json"))

    if force_rebuild and coll.count() > 0:
        _client().delete_collection(COLLECTION_NAME)
        coll = _collection()

    # Skip DOIs already indexed
    existing = set()
    if coll.count() > 0:
        got = coll.get(include=[])
        existing = set(got.get("ids", []))

    to_add: list[tuple[str, str, dict]] = []
    for f in parsed_files:
        doi = f.stem
        if doi in existing:
            continue
        d = json.loads(f.read_text())
        text = f"{d.get('title', '').strip()}\n\n{d.get('abstract', '').strip()}"
        if len(text) < 40:
            continue
        meta = {
            "title": d.get("title", "")[:400],
            "abstract_snippet": d.get("abstract", "")[:600],
            "n_pages": d.get("n_pages", 0),
        }
        to_add.append((doi, text, meta))

    if not to_add:
        return coll.count()

    # Batch add
    coll.add(
        ids=[d for d, _, _ in to_add],
        documents=[t for _, t, _ in to_add],
        metadatas=[m for _, _, m in to_add],
    )
    return coll.count()


def find_similar(
    query_text: str,
    n: int = 8,
    exclude_doi: Optional[str] = None,
) -> list[SimilarPaper]:
    """Return the top-N most similar SciPost papers to the query text."""
    coll = _collection()
    if coll.count() == 0:
        return []

    res = coll.query(
        query_texts=[query_text],
        n_results=n + (1 if exclude_doi else 0),
    )
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    out: list[SimilarPaper] = []
    for doi, doc, meta, dist in zip(ids, docs, metas, distances):
        if exclude_doi and doi == exclude_doi:
            continue
        out.append(SimilarPaper(
            doi_label=doi,
            title=meta.get("title", ""),
            similarity=max(0.0, 1.0 - float(dist)),
            abstract_snippet=meta.get("abstract_snippet", ""),
        ))
        if len(out) >= n:
            break
    return out


def find_similar_to_doi(doi: str, n: int = 8) -> list[SimilarPaper]:
    """Given a corpus DOI, return its top-N nearest neighbours in the corpus."""
    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not parsed_path.exists():
        return []
    d = json.loads(parsed_path.read_text())
    query = f"{d.get('title', '').strip()}\n\n{d.get('abstract', '').strip()}"
    return find_similar(query, n=n, exclude_doi=doi)
