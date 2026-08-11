"""Fetch specific SciPost publications by DOI label.

Bypasses the ?journal= filter (which currently silently ignores its
argument and returns everything) by hitting the direct publication
endpoint /api/publications/{doi_label}/.

Usage:
    python scripts/fetch_scipost_by_doi.py \
        --dois SciPostChem.3.1.001 SciPostChem.2.1.002 SciPostChem.2.1.001
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.ingestion.scipost import (
    API_URL,
    HEADERS,
    RATE_LIMIT_SLEEP,
    _publication_from_json,
    download_pdf,
    fetch_reports_for_publication,
    save_reports,
)


def fetch_record(doi_label: str, client: httpx.Client):
    resp = client.get(f"{API_URL}/publications/{doi_label}/")
    resp.raise_for_status()
    return _publication_from_json(resp.json())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dois", nargs="+", required=True,
                   help="SciPost DOI labels (e.g. SciPostChem.3.1.001)")
    args = p.parse_args()

    with httpx.Client(headers=HEADERS, timeout=30.0) as client:
        for doi in args.dois:
            print(f"\n=== {doi} ===")
            try:
                record = fetch_record(doi, client)
            except Exception as exc:
                print(f"  [error] metadata fetch failed: {exc}")
                continue
            print(f"  title:    {record.title[:80]}")
            print(f"  authors:  {record.author_list[:80]}")

            pdf_path = download_pdf(record)
            if pdf_path:
                print(f"  pdf:      {pdf_path}")

            try:
                reports_data = fetch_reports_for_publication(record)
                report_path = save_reports(record, reports_data)
                n = len(reports_data.get("reports", []))
                print(f"  reports:  {report_path}  ({n} report(s))")
            except Exception as exc:
                print(f"  [warn] reports fetch failed: {exc}")

            time.sleep(RATE_LIMIT_SLEEP)


if __name__ == "__main__":
    main()
