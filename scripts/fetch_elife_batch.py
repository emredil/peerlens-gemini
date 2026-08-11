"""Fetch a batch of eLife Reviewed Preprints with their eLife Assessments.

Usage:
    python scripts/fetch_elife_batch.py            # default 20
    python scripts/fetch_elife_batch.py --n 100    # custom size

Idempotent: skips PDFs and metadata already on disk.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.ingestion.elife import (
    ElifePaper,
    list_reviewed_preprints,
    download_pdf,
    save_metadata,
)
from rich.console import Console

console = Console()


def main(n: int, older_than_days: int = 60) -> None:
    console.rule(f"[bold cyan]PeerLens eLife batch: {n} reviewed preprints[/bold cyan]")

    console.print(f"\n[cyan]Step 1[/cyan] Fetching metadata from eLife API...")
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    papers: list[ElifePaper] = []
    for p in list_reviewed_preprints(max_records=n * 10):
        try:
            pub_date = datetime.fromisoformat(p.published.replace("Z", "+00:00"))
        except Exception:
            continue
        if pub_date > cutoff:
            continue
        if not (p.assessment_text or "").strip():
            continue
        papers.append(p)
        if len(papers) >= n:
            break
    console.print(f"[dim]Filtered to {len(papers)} papers older than {older_than_days} days with non-empty assessment[/dim]")
    console.print(f"[green]OK[/green] {len(papers)} papers")

    console.print(f"\n[cyan]Step 2[/cyan] Saving metadata + eLife assessments...")
    for p in papers:
        save_metadata(p)
    console.print(f"[green]OK[/green] {len(papers)} metadata files")

    console.print(f"\n[cyan]Step 3[/cyan] Downloading PDFs from bioRxiv...")
    ok = fail = skip = 0
    for p in papers:
        pdf_path = None
        try:
            pdf_path = download_pdf(p)
        except Exception as e:
            console.print(f"  [yellow]{p.doi_label}: {e}[/yellow]")
            fail += 1
            continue
        if pdf_path is None:
            console.print(f"  [yellow]{p.doi_label}: no PDF available (doi={p.doi})[/yellow]")
            fail += 1
        else:
            if pdf_path.stat().st_size < 10_000:
                console.print(f"  [yellow]{p.doi_label}: PDF suspiciously small[/yellow]")
            ok += 1
    console.print(f"[green]PDFs:[/green] ok={ok}, fail={fail}")

    console.rule("[bold green]Done[/bold green]")
    console.print(f"Metadata: data/processed/elife_reviews/")
    console.print(f"PDFs:     data/raw/pdf/elife_*.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="Number of papers to fetch")
    args = p.parse_args()
    main(args.n, older_than_days=args.older_than_days)
