"""Build the corpus index and query for a couple of papers."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.corpus_index import build_index, find_similar_to_doi
from rich.console import Console
from rich.table import Table

console = Console()

# 1. Build (or refresh) the index
console.rule("[bold]Building corpus index[/bold]")
t0 = time.time()
n = build_index()
console.print(f"[green]Indexed:[/green] {n} papers in {time.time()-t0:.1f}s")

# 2. Query — for each test DOI, get top-6 nearest neighbours
test_dois = [
    "SciPostPhys.21.1.011",  # FFD paper
    "SciPostPhys.20.3.080",  # gravitational binaries
    "SciPostPhys.20.4.097",  # axion inflation
]

for doi in test_dois:
    console.rule(f"[bold]Nearest neighbours of {doi}[/bold]")
    sims = find_similar_to_doi(doi, n=6)
    if not sims:
        console.print(f"  [yellow]No neighbours (or DOI not in corpus)[/yellow]")
        continue
    tbl = Table(show_lines=False)
    tbl.add_column("Sim", style="cyan", justify="right")
    tbl.add_column("DOI", style="magenta", max_width=25)
    tbl.add_column("Title", style="white", max_width=60)
    for s in sims:
        tbl.add_row(f"{s.similarity:.3f}", s.doi_label, s.title[:60])
    console.print(tbl)
