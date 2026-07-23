"""Parse every SciPostPhys PDF into structured JSON."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.ingestion.pdf import parse_and_save
from peerlens.settings import get_path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
import json

console = Console()

pdf_dir = get_path("raw_data") / "pdf"
processed_dir = get_path("processed_data") / "parsed"

pdfs = sorted(pdf_dir.glob("SciPostPhys.*.pdf"))
console.print(f"[cyan]Found[/cyan] {len(pdfs)} PDFs to parse")

ok = 0
fail = 0
skip = 0
failures = []

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
    console=console,
) as prog:
    task = prog.add_task("Parsing", total=len(pdfs))
    for p in pdfs:
        doi = p.stem
        target = processed_dir / f"{doi}.json"
        if target.exists():
            skip += 1
            prog.update(task, advance=1)
            continue
        try:
            parse_and_save(p, doi)
            ok += 1
        except Exception as e:
            fail += 1
            failures.append((doi, str(e)))
        prog.update(task, advance=1)

console.print(f"\n[green]Parsed:[/green] {ok}  [yellow]Skipped:[/yellow] {skip}  [red]Failed:[/red] {fail}")
if failures:
    console.print("\n[red]Failures:[/red]")
    for doi, err in failures[:10]:
        console.print(f"  {doi}: {err}")

console.print("\n[bold cyan]Corpus health check:[/bold cyan]")
stats = {
    "n": 0, "empty_abstract": 0, "no_sections": 0, "no_refs": 0,
    "n_pages_sum": 0, "eq_sum": 0, "sec_sum": 0, "ref_sum": 0,
}
for jf in sorted(processed_dir.glob("SciPostPhys.*.json")):
    d = json.loads(jf.read_text())
    stats["n"] += 1
    stats["n_pages_sum"] += d["n_pages"]
    stats["eq_sum"] += d["equation_count"]
    stats["sec_sum"] += len(d["sections"])
    stats["ref_sum"] += len(d["references"])
    if not d["abstract"]:
        stats["empty_abstract"] += 1
    if len(d["sections"]) == 0:
        stats["no_sections"] += 1
    if len(d["references"]) == 0:
        stats["no_refs"] += 1

n = max(stats["n"], 1)
console.print(f"  total parsed:      {stats['n']}")
console.print(f"  avg pages:         {stats['n_pages_sum']/n:.1f}")
console.print(f"  avg equations:     {stats['eq_sum']/n:.1f}")
console.print(f"  avg sections:      {stats['sec_sum']/n:.1f}")
console.print(f"  avg references:    {stats['ref_sum']/n:.1f}")
console.print(f"  empty abstract:    {stats['empty_abstract']}")
console.print(f"  zero sections:     {stats['no_sections']}")
console.print(f"  zero references:   {stats['no_refs']}")
