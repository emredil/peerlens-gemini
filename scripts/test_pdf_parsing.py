"""Smoke test: parse one PDF and print a structured summary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.ingestion.pdf import parse_pdf, save_parsed
from peerlens.settings import get_path
from rich.console import Console
from rich.table import Table

console = Console()

doi = "SciPostPhys.21.1.011"  # the free-fermions paper
pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"

console.print(f"[cyan]Parsing[/cyan] {pdf_path.name}")
paper = parse_pdf(pdf_path, doi)
save_parsed(paper)

console.print(f"\n[bold]Title:[/bold] {paper.title}")
console.print(f"[bold]Pages:[/bold] {paper.n_pages}")
console.print(f"[bold]Equation count:[/bold] {paper.equation_count}")
console.print(f"[bold]Abstract ({len(paper.abstract)} chars):[/bold]")
console.print(f"  {paper.abstract[:400]}...")

console.print(f"\n[bold]Sections ({len(paper.sections)}):[/bold]")
table = Table()
table.add_column("Lvl", style="cyan")
table.add_column("Num", style="yellow")
table.add_column("Title", style="magenta", max_width=40)
table.add_column("Body chars", style="green")
for s in paper.sections[:20]:
    table.add_row(str(s.level), s.number, s.title, str(len(s.text)))
console.print(table)

console.print(f"\n[bold]Figures:[/bold] {len(paper.figure_captions)}")
for c in paper.figure_captions[:3]:
    console.print(f"  - {c[:150]}")
console.print(f"\n[bold]Tables:[/bold] {len(paper.table_captions)}")

console.print(f"\n[bold]References:[/bold] {len(paper.references)}")
for r in paper.references[:3]:
    console.print(f"  - {r[:150]}")

console.print(f"\n[bold green]Saved to:[/bold green] data/processed/parsed/{doi}.json")
