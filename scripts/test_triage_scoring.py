"""Run the triage scoring engine on the 6-paper suite."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.orchestrator.scoring import score_paper, save_triage_score
from rich.console import Console
from rich.table import Table

console = Console()

DOIS = [
    "SciPostPhys.20.3.080",
    "SciPostPhys.20.4.097",
    "SciPostPhys.21.1.007",
    "SciPostPhys.20.6.166",
    "SciPostPhys.20.4.119",
    "SciPostPhys.21.1.011",
]

tbl = Table(show_lines=True, title="Triage scores")
tbl.add_column("DOI", style="cyan", max_width=25)
tbl.add_column("Score", justify="right")
tbl.add_column("Verdict", justify="right")
tbl.add_column("Math", justify="right")
tbl.add_column("Nov", justify="right")
tbl.add_column("Struct", justify="right")
tbl.add_column("Dom", justify="right")
tbl.add_column("Repro", justify="right")

for doi in DOIS:
    r = score_paper(doi)
    save_triage_score(r)
    b = r.breakdown_by_source

    verdict_color = {"priority": "green", "standard": "yellow", "reject": "red"}.get(r.verdict, "white")

    tbl.add_row(
        doi,
        str(r.score),
        f"[{verdict_color}]{r.verdict}[/{verdict_color}]",
        str(b.get("math", 0)),
        str(b.get("novelty", 0)),
        str(b.get("structure", 0)),
        str(b.get("domain", 0)),
        str(b.get("reproducibility", 0)),
    )

console.print(tbl)

console.rule("[bold]Top negative contributions (per paper)[/bold]")
for doi in DOIS:
    r = score_paper(doi)
    console.print(f"\n[cyan]{doi}[/cyan] — score {r.score}, verdict [{r.verdict}]")
    for c in r.top_negative:
        console.print(f"  {c.weight:+3d} | {c.source}.{c.kind} [{c.severity}] @ {c.location}",
                      markup=False)
        console.print(f"        {c.evidence_short[:100]}", markup=False)
