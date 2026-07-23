"""Extend the corpus with more SciPost Physics papers, filtered by prefix.

Idempotent: skips PDFs/reports already on disk.
Filters strictly to SciPostPhys.* (excludes Core, Codeb, LectNotes, CommRep).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
from peerlens.ingestion.scipost import (
    HEADERS, API_URL, RATE_LIMIT_SLEEP,
    _publication_from_json, download_pdf,
    fetch_reports_for_publication, save_reports,
)
from peerlens.settings import get_path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


def is_target_paper(doi_label: str) -> bool:
    """Only main SciPost Physics journal, not sister journals."""
    return (
        doi_label.startswith("SciPostPhys.")
        and not any(x in doi_label for x in ["Codeb", "LectNotes", "Proc", "CommRep"])
    )


def existing_dois() -> set[str]:
    """DOIs we've already fetched."""
    result = set()
    for meta_file in (get_path("raw_data") / "metadata").glob("*.jsonl"):
        with meta_file.open() as f:
            for line in f:
                r = json.loads(line)
                result.add(r["doi_label"])
    return result


def stream_publications(journal: str, limit_per_page: int = 100):
    """Yield paginated publication records."""
    url = f"{API_URL}/publications/?journal={journal}&limit={limit_per_page}"
    with httpx.Client(headers=HEADERS, timeout=30.0) as client:
        while url:
            resp = client.get(url)
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("results", []):
                yield item
            url = payload.get("next")
            time.sleep(RATE_LIMIT_SLEEP)


def main(target_new: int, journal: str) -> None:
    console.rule(f"[bold cyan]Extending corpus with {target_new} new {journal} papers[/bold cyan]")
    already = existing_dois()
    console.print(f"[dim]Already have {len(already)} records on disk[/dim]")

    new_records = []
    for item in stream_publications(journal):
        doi = item.get("doi_label", "")
        if not is_target_paper(doi):
            continue
        if doi in already:
            continue
        new_records.append(_publication_from_json(item))
        if len(new_records) >= target_new:
            break

    console.print(f"[green]Collected[/green] {len(new_records)} new SciPostPhys.* records")

    # Save metadata
    out = get_path("raw_data") / "metadata" / f"SciPostPhys_extension_{len(new_records)}.jsonl"
    from dataclasses import asdict
    with out.open("w", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    console.print(f"[green]Saved[/green] metadata to {out}")

    # PDFs
    console.print("\n[cyan]Downloading PDFs...[/cyan]")
    pdf_ok = 0
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("PDFs", total=len(new_records))
        for r in new_records:
            if download_pdf(r) is not None:
                pdf_ok += 1
            prog.update(task, advance=1)
    console.print(f"[green]PDFs downloaded:[/green] {pdf_ok}/{len(new_records)}")

    # Reports
    console.print("\n[cyan]Fetching referee reports...[/cyan]")
    rep_ok = 0
    total_reports = 0
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("Reports", total=len(new_records))
        for r in new_records:
            try:
                data = fetch_reports_for_publication(r)
                save_reports(r, data)
                n = sum(len(v) for v in data["versions"].values())
                total_reports += n
                rep_ok += 1
            except Exception as e:
                console.print(f"[red]Failed[/red] {r.doi_label}: {e}")
            prog.update(task, advance=1)

    console.print(f"[green]Reports fetched:[/green] {rep_ok}/{len(new_records)}, total individual reports: {total_reports}")
    console.rule("[bold green]Extension complete[/bold green]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--journal", default="SciPostPhys")
    args = p.parse_args()
    main(args.n, args.journal)
