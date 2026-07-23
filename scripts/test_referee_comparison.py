"""Compare our AI math report against the real referee reports for one paper."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.evaluation.referee_comparison import compare, save_comparison
from rich.console import Console
from rich.table import Table

console = Console()
doi = "SciPostPhys.21.1.011"

result = compare(doi)
path = save_comparison(result)

console.rule(f"[bold cyan]Comparison — {doi}[/bold cyan]")
console.print(f"[dim]Model: {result.model}, tokens in/out: {result.input_tokens}/{result.output_tokens}[/dim]")
console.print()
console.print(f"AI findings:              {result.n_ai_findings}")
console.print(f"Referee math concerns:    {result.n_referee_math_concerns}")
console.print(f"Matches:                  {result.n_matches}")
console.print(f"Missed by AI:             {result.n_missed}")
console.print(f"Novel (not in referees):  {result.n_novel}")
console.print()
console.rule("[bold]Summary[/bold]")
console.print(result.summary, markup=False)

if result.matches:
    console.rule("[bold]Matches[/bold]")
    for m in result.matches:
        console.print(f"[green]•[/green] AI at {m.get('ai_finding_location', '')} "
                      f"↔ {m.get('referee_concern_ref', '')} ({m.get('match_quality', '')})",
                      markup=False)
        if m.get("notes"):
            console.print(f"    {m['notes']}", markup=False)

if result.missed:
    console.rule("[bold red]Missed by AI[/bold red]")
    for c in result.missed:
        console.print(f"[red]•[/red] {c.referee_id}: {c.concern}", markup=False)

if result.novel:
    console.rule("[bold yellow]Novel AI findings (not in any referee)[/bold yellow]")
    for n in result.novel:
        console.print(f"[yellow]•[/yellow] {n.get('ai_finding_location', '')} — "
                      f"{n.get('assessment', '')}", markup=False)
        if n.get("rationale"):
            console.print(f"    {n['rationale']}", markup=False)

console.print()
console.print(f"[dim]Saved: {path}[/dim]")

# cost
cost = (result.input_tokens/1e6)*3.00 + (result.output_tokens/1e6)*15.00
console.print(f"[dim]Estimated cost: ${cost:.4f}[/dim]")
