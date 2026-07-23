import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.structure_agent import analyze_structure, save_report, scan_sections
from peerlens.settings import get_path
import json
from rich.console import Console
from rich.table import Table

console = Console()

doi = "SciPostPhys.21.1.011"

parsed = json.loads((get_path("processed_data") / "parsed" / f"{doi}.json").read_text())
cands = scan_sections(parsed)

console.rule("[bold]Stage 1: pattern scan[/bold]")
console.print(f"Candidates found: [green]{len(cands)}[/green]")
if cands:
    for c in cands[:6]:
        console.print(f"  Section {c.section_number}: [yellow]{c.trigger_phrase!r}[/yellow]",
                      markup=False)

console.rule("[bold]Stage 2: LLM adjudication[/bold]")
report = analyze_structure(doi)
save_report(report)

console.print(f"Findings: [green]{report.n_findings}[/green] "
              f"(crit={report.n_critical}, warn={report.n_warning}, info={report.n_info})")
console.print(f"[dim]Model: {report.model}, tokens {report.input_tokens}/{report.output_tokens}[/dim]")

if report.findings:
    tbl = Table(show_lines=True)
    tbl.add_column("Kind", style="cyan", max_width=18)
    tbl.add_column("Sev", style="red", max_width=8)
    tbl.add_column("Location", style="yellow", max_width=20)
    tbl.add_column("Trigger", style="magenta", max_width=25)
    tbl.add_column("Evidence", style="white", max_width=45)
    for f in report.findings:
        tbl.add_row(f.kind, f.severity, f.location, f.trigger_phrase[:25],
                    f.evidence[:200])
    console.print(tbl)

cost = (report.input_tokens/1e6)*0.075 + (report.output_tokens/1e6)*0.30
console.print(f"\n[dim]Estimated cost: ${cost:.4f}[/dim]")
