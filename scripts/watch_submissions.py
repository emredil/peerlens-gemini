"""Submission folder watcher — placeholder for journal integration.

Concept
-------
In production this would be wired to a specific journal's Editorial
Manager / ScholarOne / OJS / SFTP endpoint. As a portable, journal-
agnostic placeholder, we watch a filesystem folder: any PDF that lands
there is treated as a new submission and pushed through the PeerLens
pipeline automatically.

At sale time this script is replaced (or fronted by) an adapter for the
customer journal's actual submission channel — webhook receiver, API
poller, or S3/SFTP consumer.

Usage
-----
    python scripts/watch_submissions.py \\
        --incoming ~/peerlens/incoming \\
        --archive ~/peerlens/processed \\
        --interval 10

Layout
------
    incoming/    new submissions dropped here (by the journal's system)
    archive/     PDFs are moved here after successful analysis
    incoming/failed/    PDFs that failed are moved here

The script runs until interrupted with Ctrl+C.
"""
import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()


def sanitize_stem(stem: str) -> str:
    """Turn a filename stem into a safe doi_label."""
    return "".join(c if c.isalnum() or c in ".-_" else "_" for c in stem) or "manuscript"


def process_one(pdf: Path, archive: Path, failed: Path) -> bool:
    """Run the full pipeline for a single PDF. Return True on success."""
    doi_label = sanitize_stem(pdf.stem)
    console.print(f"[cyan]{datetime.now().strftime('%H:%M:%S')}[/cyan] "
                  f"processing [white]{pdf.name}[/white] as [magenta]{doi_label}[/magenta]")

    project_root = Path(__file__).resolve().parents[1]
    analyze_script = project_root / "scripts" / "analyze_paper.py"

    proc = subprocess.run(
        [sys.executable, str(analyze_script), "--pdf", str(pdf), "--doi", doi_label],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        console.print(f"  [red]FAILED[/red] returncode={proc.returncode}")
        console.print(f"  [dim]{proc.stderr[-500:]}[/dim]")
        failed.mkdir(parents=True, exist_ok=True)
        target = failed / pdf.name
        if target.exists():
            target = failed / f"{pdf.stem}_{int(time.time())}.pdf"
        shutil.move(str(pdf), str(target))
        return False

    # Extract the last "Triage:" line from the analyze_paper output
    verdict_line = ""
    for line in proc.stdout.splitlines():
        if "Triage:" in line:
            verdict_line = line.strip()
    if verdict_line:
        console.print(f"  → {verdict_line}")

    archive.mkdir(parents=True, exist_ok=True)
    target = archive / pdf.name
    if target.exists():
        target = archive / f"{pdf.stem}_{int(time.time())}.pdf"
    shutil.move(str(pdf), str(target))
    console.print(f"  [green]archived[/green] -> {target}")
    return True


def watch(incoming: Path, archive: Path, interval: int) -> None:
    incoming.mkdir(parents=True, exist_ok=True)
    failed = incoming / "failed"

    console.rule("[bold cyan]PeerLens submission watcher[/bold cyan]")
    console.print(f"Watching:   {incoming}")
    console.print(f"Archive:    {archive}")
    console.print(f"Failed:     {failed}")
    console.print(f"Poll:       every {interval}s")
    console.print(f"[dim]Ctrl+C to stop.[/dim]")

    seen: set[str] = set()

    while True:
        try:
            pdfs = sorted(
                p for p in incoming.glob("*.pdf")
                if p.is_file() and p.parent == incoming
            )
            new_pdfs = [p for p in pdfs if p.name not in seen]
            for pdf in new_pdfs:
                seen.add(pdf.name)
                # Small settle delay in case the file is still being written
                time.sleep(1.0)
                process_one(pdf, archive, failed)
            time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped by user.[/yellow]")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incoming", type=Path, required=True,
                    help="Folder to watch for new PDFs")
    ap.add_argument("--archive", type=Path, required=True,
                    help="Where processed PDFs are moved")
    ap.add_argument("--interval", type=int, default=10,
                    help="Polling interval in seconds (default: 10)")
    args = ap.parse_args()
    watch(args.incoming.expanduser(), args.archive.expanduser(), args.interval)


if __name__ == "__main__":
    main()
