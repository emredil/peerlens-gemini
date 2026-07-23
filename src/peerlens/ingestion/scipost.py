"""SciPost Physics ingestion client.

Fetches publication metadata, PDFs, and referee reports from the public
SciPost REST API and HTML pages.

Endpoints used:
  - GET /api/publications/?journal=SciPostPhys  (paginated JSON list)
  - GET {publication_url}/pdf                    (binary PDF)
  - GET {submission_url}                         (HTML page with reports)

Referee reports are HTML-only; parsing is delegated to report_parser.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import httpx
from rich.console import Console
from rich.progress import track

from peerlens.settings import CONFIG, get_path
from peerlens.ingestion.report_parser import parse_reports_from_html, PeerReport

console = Console()

BASE_URL = CONFIG["corpus"]["base_url"].rstrip("/")
API_URL = CONFIG["corpus"]["api_url"].rstrip("/")
RATE_LIMIT_SLEEP = 0.5

USER_AGENT = "PeerLens/0.0.1 (research prototype; contact: research@peerlens.local)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}


# =========================================================================
# Publication metadata
# =========================================================================

@dataclass
class PublicationRecord:
    """A single SciPost publication record."""
    doi_label: str
    doi: str
    title: str
    author_list: str
    abstract: str
    submission_date: str
    acceptance_date: str
    publication_date: str
    specialties: list[str]
    topics: list[str]
    approaches: str
    publication_url: str
    submission_url: str | None
    thread_hash: str | None
    citation: str
    cc_license: str


def _publication_from_json(item: dict) -> PublicationRecord:
    accepted = item.get("accepted_submission") or {}
    return PublicationRecord(
        doi_label=item.get("doi_label", ""),
        doi=item.get("doi", ""),
        title=item.get("title", ""),
        author_list=item.get("author_list", ""),
        abstract=item.get("abstract", ""),
        submission_date=item.get("submission_date", ""),
        acceptance_date=item.get("acceptance_date", ""),
        publication_date=item.get("publication_date", ""),
        specialties=item.get("specialties", []) or [],
        topics=item.get("topics", []) or [],
        approaches=item.get("approaches", "") or "",
        publication_url=item.get("url", ""),
        submission_url=accepted.get("url"),
        thread_hash=accepted.get("thread_hash"),
        citation=item.get("citation", ""),
        cc_license=item.get("cc_license", ""),
    )


def list_publications(
    journal: str = "SciPostPhys",
    limit: int = 50,
    max_records: int | None = None,
) -> Iterator[PublicationRecord]:
    """Yield publication records, paginating through the API."""
    url = f"{API_URL}/publications/?journal={journal}&limit={limit}"
    fetched = 0

    with httpx.Client(headers=HEADERS, timeout=30.0) as client:
        while url:
            resp = client.get(url)
            resp.raise_for_status()
            payload = resp.json()

            for item in payload.get("results", []):
                yield _publication_from_json(item)
                fetched += 1
                if max_records is not None and fetched >= max_records:
                    return

            url = payload.get("next")
            time.sleep(RATE_LIMIT_SLEEP)


def save_metadata(records: list[PublicationRecord], filename: str = "metadata.jsonl") -> Path:
    out_dir = get_path("raw_data") / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    console.print(f"[green]Saved[/green] {len(records)} records to {out_path}")
    return out_path


# =========================================================================
# PDF download
# =========================================================================

def download_pdf(record: PublicationRecord, force: bool = False) -> Path | None:
    pdf_dir = get_path("raw_data") / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_path = pdf_dir / f"{record.doi_label}.pdf"

    if out_path.exists() and not force:
        return out_path

    pdf_url = f"{BASE_URL}{record.publication_url}/pdf"
    try:
        with httpx.Client(headers=HEADERS, timeout=60.0, follow_redirects=True) as client:
            resp = client.get(pdf_url)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF-"):
                console.print(f"[yellow]Not a PDF[/yellow] for {record.doi_label}")
                return None
            out_path.write_bytes(resp.content)
            time.sleep(RATE_LIMIT_SLEEP)
            return out_path
    except httpx.HTTPError as e:
        console.print(f"[red]PDF download failed[/red] for {record.doi_label}: {e}")
        return None


def fetch_batch(journal: str = "SciPostPhys", n: int = 10, download_pdfs: bool = True) -> list[PublicationRecord]:
    console.print(f"[cyan]Fetching[/cyan] up to {n} publications from {journal}...")
    records = list(list_publications(journal=journal, max_records=n))
    save_metadata(records, filename=f"{journal}_first_{n}.jsonl")

    if download_pdfs:
        console.print(f"[cyan]Downloading[/cyan] {len(records)} PDFs...")
        successful = 0
        for r in track(records, description="PDFs"):
            if download_pdf(r) is not None:
                successful += 1
        console.print(f"[green]Downloaded[/green] {successful}/{len(records)} PDFs")
    return records


# =========================================================================
# Referee reports
# =========================================================================

def _parse_submission_id(url: str) -> tuple[str, int]:
    m = re.search(r"(scipost_\d+_\d+)v(\d+)", url)
    if not m:
        return "", 0
    return m.group(1), int(m.group(2))


def discover_all_versions(submission_url: str) -> list[str]:
    """Find all versions (v1, v2, ...) of a submission thread."""
    with httpx.Client(headers=HEADERS, timeout=30.0) as client:
        resp = client.get(f"{BASE_URL}{submission_url}")
        resp.raise_for_status()
        html = resp.text

    base_id, _ = _parse_submission_id(submission_url)
    if not base_id:
        return [submission_url]

    pattern = rf"/submissions/{base_id}v(\d+)/"
    versions = sorted({int(v) for v in re.findall(pattern, html)})
    return [f"/submissions/{base_id}v{v}/" for v in versions]


def fetch_reports_for_publication(record: PublicationRecord) -> dict:
    """Fetch all reports across all submission versions for a publication."""
    if not record.submission_url:
        return {"submission_id": "", "versions": {}}

    version_urls = discover_all_versions(record.submission_url)
    all_versions: dict[int, list[PeerReport]] = {}

    with httpx.Client(headers=HEADERS, timeout=30.0) as client:
        for vurl in version_urls:
            base_id, version = _parse_submission_id(vurl)
            resp = client.get(f"{BASE_URL}{vurl}")
            if resp.status_code != 200:
                console.print(f"[yellow]Skipping {vurl}: HTTP {resp.status_code}[/yellow]")
                continue
            reports = parse_reports_from_html(resp.text, base_id, version)
            all_versions[version] = reports
            time.sleep(RATE_LIMIT_SLEEP)

    base_id, _ = _parse_submission_id(record.submission_url)
    return {"submission_id": base_id, "versions": all_versions}


def save_reports(record: PublicationRecord, reports_data: dict) -> Path:
    out_dir = get_path("raw_data") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record.doi_label}.json"

    serializable = {
        "doi_label": record.doi_label,
        "submission_id": reports_data["submission_id"],
        "versions": {
            str(v): [asdict(r) for r in reports]
            for v, reports in reports_data["versions"].items()
        }
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return out_path
