"""Fetch a real batch of SciPost Physics publications with reports.

Usage:
    python scripts/fetch_corpus_batch.py            # default 50
    python scripts/fetch_corpus_batch.py --n 100    # custom size

Idempotent: skips PDFs and reports that already exist on disk.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.ingestion.scipost import (
    PublicationRecord,
    list_publications,
    save_metadata,
    download_pdf,
    fetch_reports_for_publication,
    save_reports,
)
from peerlens.settings import get_path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


def main(n: int, journal: str) -> None:
    console.rule(f"[bold cyan]PeerLens corpus batch: {n} papers from {journal}[/bold cyan]")

    # Step 1: metadata
    console.print(f"\n[cyan]Step 1[/cyan] Fetching metadata...")
    records = list(list_publications(journal=journal, max_records=n))
    save_metadata(records, filename=f"{journal}_batch_{n}.jsonl")
    console.print(f"[green]OK[/green] {len(records)} records")

    # Step 2: PDFs
    console.print(f"\n[cyan]Step 2[/cyan] Downloading PDFs...")
    pdf_dir = get_path("raw_data") / "pdf"
    pdf_ok = 0
    pdf_skip = 0
    pdf_fail = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("PDFs", total=len(records))
        for r in records:
            target = pdf_dir / f"{r.doi_label}.pdf"
            if target.exists():
                pdf_skip += 1
            else:
                path = download_pdf(r)
                if path is not None:
                    pdf_ok += 1
                else:
                    pdf_fail += 1
            prog.update(task, advance=1)

    console.print(f"[green]OK[/green] downloaded={pdf_ok} skipped={pdf_skip} failed={pdf_fail}")

    # Step 3: reports
    console.print(f"\n[cyan]Step 3[/cyan] Fetching referee reports...")
    reports_dir = get_path("raw_data") / "reports"
    rep_ok = 0
    rep_skip = 0
    rep_fail = 0
    total_reports = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Reports", total=len(records))
        for r in records:
            target = reports_dir / f"{r.doi_label}.json"
            if target.exists():
                rep_skip += 1
                with target.open() as f:
                    d = json.load(f)
                total_reports += sum(len(v) for v in d["versions"].values())
            else:
                try:
                    data = fetch_reports_for_publication(r)
                    save_reports(r, data)
                    n_rep = sum(len(v) for v in data["versions"].values())
                    total_reports += n_rep
                    rep_ok += 1
                except Exception as e:
                    console.print(f"[red]Reports failed[/red] for {r.doi_label}: {e}")
                    rep_fail += 1
            prog.update(task, advance=1)

    console.print(f"[green]OK[/green] fetched={rep_ok} skipped={rep_skip} failed={rep_fail}")
    console.print(f"[bold]Total referee reports across corpus: {total_reports}[/bold]")

    # Summary
    console.rule("[bold green]Batch complete[/bold green]")
    console.print(f"Metadata:  {len(records)} records")
    console.print(f"PDFs:      {pdf_ok + pdf_skip} on disk ({pdf_fail} failed)")
    console.print(f"Reports:   {rep_ok + rep_skip} publications, {total_reports} individual reports")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--journal", default="SciPostPhys")
    args = p.parse_args()
    main(args.n, args.journal)
