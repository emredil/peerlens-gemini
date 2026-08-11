"""Fetch a batch of F1000Research papers with completed peer reviews."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

from peerlens.ingestion.f1000 import (
    RATE_LIMIT_SLEEP,
    USER_AGENT,
    fetch_paper,
    iter_domain_papers,
    save_paper_bundle,
)
from peerlens.settings import get_path


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "*/*"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="Mathematics",
                   help="F1000 R_SUB filter (Mathematics / Chemistry / ...)")
    p.add_argument("--domain", default="mathematics",
                   help="Filesystem tag used under data/f1000/<domain>/")
    p.add_argument("--n", type=int, default=3, help="Number of papers to fetch")
    p.add_argument("--dois", nargs="*", default=None,
                   help="Explicit DOI list; overrides subject search")
    args = p.parse_args()

    if args.dois:
        papers = []
        with httpx.Client(timeout=60, headers=_headers()) as client:
            for doi in args.dois:
                paper = fetch_paper(doi, domain=args.domain, client=client)
                papers.append(paper)
                time.sleep(RATE_LIMIT_SLEEP)
    else:
        papers = iter_domain_papers(
            subject=args.subject,
            domain_label=args.domain,
            max_papers=args.n,
        )

    print(f"\nFetched {len(papers)} papers with completed reviews\n")
    for paper in papers:
        print(f"-> {paper.doi_label}  |  {paper.verdict}  |  {paper.title[:70]}")
        paths = save_paper_bundle(paper)
        print(f"    pdf:      {paths['pdf']}")
        print(f"    metadata: {paths['metadata']}")
        print(f"    reviews:  {paths['reviews']}\n")


if __name__ == "__main__":
    main()
