"""Smoke test the SymPy verifier on one paper.

Runs up to 5 sympy-tagged claims from SciPostPhys.21.1.011 to keep
cost/time bounded while validating the pipeline end to end.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.sympy_verifier import verify_paper, save_report
from peerlens.settings import get_path
from rich.console import Console
from rich.table import Table

console = Console()

doi = "SciPostPhys.21.1.011"
claims_path = get_path("processed_data") / "claims" / f"{doi}.json"
claims_data = json.loads(claims_path.read_text())

n_sympy = sum(1 for c in claims_data["claims"] if c.get("verification_method") == "sympy")
console.print(f"[cyan]Verifying[/cyan] {doi}  (sympy-tagged claims: {n_sympy})")

pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
report = verify_paper(claims_data, pdf_path, max_claims=5)
save_report(report)

console.rule("[bold]Summary[/bold]")
console.print(f"Verified:          {report.n_verified}")
console.print(f"Refuted:           {report.n_refuted}")
console.print(f"Unable to verify:  {report.n_unable}")
console.print(f"[dim]Tokens in/out: {report.total_tokens_in}/{report.total_tokens_out}[/dim]")

console.rule("[bold]Findings[/bold]")
tbl = Table(show_lines=True)
tbl.add_column("Loc", style="yellow", max_width=14)
tbl.add_column("Type", style="cyan", max_width=12)
tbl.add_column("Verdict", style="bold", max_width=10)
tbl.add_column("Sev", style="red", max_width=6)
tbl.add_column("Time", style="magenta", max_width=6)
tbl.add_column("Detail", style="dim", max_width=50)
for f in report.findings:
    tbl.add_row(
        f.location[:14],
        f.claim_type[:12],
        f.verdict,
        f.severity,
        f"{f.runtime_seconds:.2f}s",
        f.detail[:200],
    )
console.print(tbl)

console.rule("[bold]Generated snippets (first 3, abbreviated)[/bold]")
for f in report.findings[:3]:
    console.print(f"\n[yellow]{f.location} — {f.verdict}[/yellow]")
    console.print(f.generated_code[:600])
    if len(f.generated_code) > 600:
        console.print("[dim]... (truncated)[/dim]")

# cost
cost = (report.total_tokens_in/1e6)*0.80 + (report.total_tokens_out/1e6)*4.00
console.print(f"\n[dim]Estimated cost: ${cost:.4f}[/dim]")
