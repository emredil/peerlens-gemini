"""Aggregate math + novelty for all pipeline papers, print a compact dashboard."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.orchestrator.referee_aggregator import aggregate_and_render_full
from peerlens.settings import get_path
from rich.console import Console
from rich.table import Table

console = Console()

DOIS = [
    "SciPostPhys.20.3.080",
    "SciPostPhys.20.4.097",
    "SciPostPhys.21.1.007",
    "SciPostPhys.20.6.166",
    "SciPostPhys.20.4.119",
    "SciPostPhys.21.1.011",  # the hero paper, for completeness
]

tbl = Table(title="Full Referee Reports")
tbl.add_column("DOI", style="cyan")
tbl.add_column("Overall", style="bold")
tbl.add_column("Math", style="magenta")
tbl.add_column("Novelty", style="yellow")
tbl.add_column("Reason", style="dim", max_width=45)

for doi in DOIS:
    try:
        json_path, md_path = aggregate_and_render_full(doi)
        import json
        d = json.loads(json_path.read_text())
        tbl.add_row(
            doi,
            d["overall_verdict"],
            d["math_verdict"],
            d["novelty_verdict"],
            d["overall_reason"][:200],
        )
    except Exception as e:
        tbl.add_row(doi, "ERROR", "-", "-", f"{type(e).__name__}: {str(e)[:80]}")

console.print(tbl)
console.print(f"\n[dim]Reports in data/processed/referee_reports/, "
              f"markdown in data/processed/referee_sections/*_full.md[/dim]")
