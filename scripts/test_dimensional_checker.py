import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.dimensional_checker import check_paper, save_report
from peerlens.settings import get_path
from rich.console import Console
from rich.table import Table

console = Console()

doi = "SciPostPhys.21.1.011"
claims_path = get_path("processed_data") / "claims" / f"{doi}.json"
pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"

if not claims_path.exists():
    console.print(f"[red]Missing claims JSON for {doi}. Run claim extractor first.[/red]")
    sys.exit(1)
if not pdf_path.exists():
    console.print(f"[red]Missing PDF: {pdf_path}[/red]")
    sys.exit(1)

claims_data = json.loads(claims_path.read_text())
console.print(f"[cyan]Checking[/cyan] {doi}")
console.print(f"  claims: {len(claims_data.get('claims', []))}, symbols: {len(claims_data.get('symbols', []))}")

report = check_paper(claims_data, pdf_path, max_vision_calls=3)
save_report(report)

console.rule("[bold]Summary[/bold]")
console.print(f"Total findings:    {report.n_findings}")
console.print(f"  critical:        {report.n_critical}")
console.print(f"  warning:         {report.n_warning}")
console.print(f"  info:            {report.n_info}")
console.print(f"[dim]Tier A tokens in/out: {report.tier_a_tokens_in}/{report.tier_a_tokens_out}[/dim]")
console.print(f"[dim]Tier B tokens in/out: {report.tier_b_tokens_in}/{report.tier_b_tokens_out}, "
              f"pages analyzed: {report.tier_b_pages_analyzed}[/dim]")

console.rule("[bold]Findings[/bold]")
tbl = Table(show_lines=True)
tbl.add_column("Loc", style="yellow", max_width=14)
tbl.add_column("Verdict", style="cyan", max_width=14)
tbl.add_column("Sev", style="red", max_width=8)
tbl.add_column("Tier", style="magenta", max_width=8)
tbl.add_column("Evidence", style="white", max_width=45)
tbl.add_column("Fix", style="dim", max_width=30)
tbl.add_column("Conf", style="green", max_width=5)
for f in report.findings:
    tbl.add_row(
        f.location, f.verdict, f.severity, f.verifier_tier,
        f.evidence, f.suggested_fix or "-", f"{f.confidence:.2f}",
    )
console.print(tbl)
