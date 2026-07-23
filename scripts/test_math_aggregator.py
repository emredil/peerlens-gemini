"""Aggregate the dimensional + sympy findings for one paper and preview the
resulting Math Referee Section."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.orchestrator.math_aggregator import (
    aggregate, save_math_report, render_markdown, save_markdown,
)
from rich.console import Console

console = Console()

doi = "SciPostPhys.21.1.011"

report = aggregate(doi)
json_path = save_math_report(report)
md_path = save_markdown(report)

console.rule(f"[bold cyan]Math Report — {doi}[/bold cyan]")
console.print(f"[bold]Verdict:[/bold] [magenta]{report.verdict}[/magenta]")
console.print(f"[dim]{report.verdict_reason}[/dim]")
console.print()
console.print(f"Claims extracted:       {report.n_claims_extracted}")
console.print(f"Dimensional checked:    {report.n_dimensional_checked} "
              f"(crit {report.n_dimensional_critical}, warn {report.n_dimensional_warning})")
console.print(f"Symbolic attempted:     {report.n_sympy_tagged}")
console.print(f"  verified:             {report.n_sympy_verified}")
console.print(f"  refuted:              {report.n_sympy_refuted}")
console.print(f"  unable_to_verify:     {report.n_sympy_unable}")
console.print(f"Paraphrase mismatches:  {report.n_paraphrase_mismatch}")
console.print(f"Coverage confidence:    {report.coverage_confidence}")
console.print()
console.print(f"[green]JSON:[/green] {json_path}")
console.print(f"[green]MD:[/green]   {md_path}")

console.rule("[bold]Markdown preview (first 60 lines)[/bold]")
md = md_path.read_text()
for line in md.splitlines()[:60]:
    console.print(line)
