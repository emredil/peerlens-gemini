"""Verify Semantic Scholar API works for our use case."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.semantic_scholar import (
    search_by_query, get_paper, get_references, get_recent_related,
)
from rich.console import Console
from rich.table import Table

console = Console()

# --- Test 1: search by query ---
console.rule("[bold]Search: FFD spin chain integrable models[/bold]")
results = search_by_query("free fermions in disguise integrable spin chain", limit=5, year_from=2020)
for r in results:
    console.print(f"  [{r.year}] {r.title[:80]}",
                  f"— cited by {r.citation_count}", markup=False)
    if r.external_ids:
        console.print(f"      ids: {r.external_ids}", markup=False)

# --- Test 2: get a specific paper by DOI ---
console.rule("[bold]Get paper by DOI (SciPostPhys.21.1.011)[/bold]")
paper = get_paper("DOI:10.21468/SciPostPhys.21.1.013")
if paper:
    console.print(f"  Found: {paper.title[:80]}", markup=False)
    console.print(f"  Year: {paper.year}, refs: {paper.reference_count}")
    console.print(f"  paperId: {paper.paper_id}")
else:
    console.print("  [red]Not found via DOI lookup[/red]")

# --- Test 3: recent related papers ---
if paper and paper.paper_id:
    console.rule("[bold]Related papers (2023+)[/bold]")
    related = get_recent_related(paper.paper_id, limit=10, year_from=2023)
    tbl = Table(show_lines=False)
    tbl.add_column("Year", style="cyan")
    tbl.add_column("Title", style="white", max_width=60)
    tbl.add_column("Venue", style="magenta", max_width=25)
    tbl.add_column("Cited", style="green")
    for r in related:
        tbl.add_row(str(r.year or "?"), r.title[:60], r.venue[:25], str(r.citation_count))
    console.print(tbl)
