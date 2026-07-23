import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.domain_expert import analyze_domain, save_report
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
    report = analyze_domain(doi)
    save_report(report)

    console.print(f"Subdomain: [cyan]{report.subdomain_used}[/cyan] | Persona: {report.persona}")
    console.print(f"Findings: {report.n_findings} "
                  f"(crit={report.n_critical}, warn={report.n_warning}, info={report.n_info})")
    cost = (report.input_tokens/1e6)*0.075 + (report.output_tokens/1e6)*0.30
    console.print(f"[dim]tokens {report.input_tokens}/{report.output_tokens}, cost ${cost:.4f}[/dim]")

    if report.findings:
        tbl = Table(show_lines=False)
        tbl.add_column("Kind", style="cyan", max_width=18)
        tbl.add_column("Sev", style="red", max_width=8)
        tbl.add_column("Loc", style="yellow", max_width=15)
        tbl.add_column("Evidence", style="white", max_width=70)
        for f in report.findings:
            tbl.add_row(f.kind, f.severity, f.location, f.evidence[:200])
        console.print(tbl)
