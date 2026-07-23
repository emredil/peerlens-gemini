"""Run the novelty agent on our hero FFD paper and print structured output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.novelty_agent import analyze_novelty, save_report
from rich.console import Console
from rich.table import Table

console = Console()

doi = "SciPostPhys.21.1.011"

console.print(f"[cyan]Analyzing novelty for[/cyan] {doi}")
report = analyze_novelty(doi)
path = save_report(report)

console.rule("[bold]Retrieval summary[/bold]")
console.print(f"Related work retrieved:    {report.n_related_retrieved}")
console.print(f"  from corpus:             {report.n_corpus_matches}")
console.print(f"  from Semantic Scholar:   {report.n_scholar_matches}")
console.print(f"Uncited but close:         {report.n_uncited_but_close}")
console.print(f"Reference year median:     {report.reference_year_median}")
console.print(f"Reference year max:        {report.reference_year_max}")

console.rule("[bold]Findings[/bold]")
tbl = Table(show_lines=True)
tbl.add_column("Kind", style="cyan", max_width=16)
tbl.add_column("Sev", style="red", max_width=8)
tbl.add_column("Loc", style="yellow", max_width=16)
tbl.add_column("Evidence", style="white", max_width=55)
tbl.add_column("Conf", style="green", max_width=6)
for f in report.findings:
    tbl.add_row(
        f.kind, f.severity, f.location or "-",
        f.evidence[:220], f"{f.confidence:.2f}",
    )
console.print(tbl)

if report.findings:
    console.rule("[bold]Detailed reasoning per finding[/bold]")
    for i, f in enumerate(report.findings):
        console.print(f"\n[bold cyan][{i}] {f.kind} — {f.severity}[/bold cyan]", markup=False)
        console.print(f"  Location: {f.location}", markup=False)
        console.print(f"  Evidence: {f.evidence}", markup=False)
        if f.suggested_fix:
            console.print(f"  Suggested fix: {f.suggested_fix}", markup=False)
        if f.references:
            console.print(f"  Referenced related work:", markup=False)
            for r in f.references[:5]:
                console.print(f"    - {r}", markup=False)

# cost
cost = (report.input_tokens/1e6)*3.00 + (report.output_tokens/1e6)*15.00
console.print(f"\n[dim]Model: {report.model}, tokens {report.input_tokens}/{report.output_tokens}, cost: ${cost:.4f}[/dim]")
