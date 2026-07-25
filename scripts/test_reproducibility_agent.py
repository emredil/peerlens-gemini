import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.reproducibility_agent import analyze_reproducibility, save_report
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

for doi in DOIS:
    console.rule(f"[bold]{doi}[/bold]")
    report = analyze_reproducibility(doi)
    save_report(report)

    console.print(f"Paper types: {report.paper_types}")
    console.print(f"Scrutiny: [cyan]{report.scrutiny_level}[/cyan]  reason: {report.gate_reason}")
    console.print(f"Signals: URLs={len(report.detected_signals.get('code_urls', []))}, "
                  f"algo={len(report.detected_signals.get('algorithms_mentioned', []))}")
    console.print(f"Findings: [green]{report.n_findings}[/green] "
                  f"(crit={report.n_critical}, warn={report.n_warning})")
    if report.findings:
        tbl = Table(show_lines=False)
        tbl.add_column("Kind", style="cyan", max_width=20)
        tbl.add_column("Sev", style="red", max_width=8)
        tbl.add_column("Loc", style="yellow", max_width=20)
        tbl.add_column("Evidence", style="white", max_width=60)
        for f in report.findings:
            tbl.add_row(f.kind, f.severity, f.location, f.evidence[:200])
        console.print(tbl)
