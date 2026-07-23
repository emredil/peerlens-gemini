"""Test the equation extractor on one paper: pull equations for the paper's
sympy-tagged claims and print them."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.equation_extractor import extract_equations_for_claim
from peerlens.settings import get_path
from rich.console import Console
from rich.table import Table

console = Console()

doi = "SciPostPhys.21.1.011"
pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
claims_path = get_path("processed_data") / "claims" / f"{doi}.json"

data = json.loads(claims_path.read_text())

# Collect distinct equation numbers referenced by sympy-tagged claims
from peerlens.agents.equation_extractor import _normalize_eq_number

wanted = set()
for c in data["claims"]:
    if c.get("verification_method") == "sympy":
        for eq in c.get("referenced_equations", []):
            wanted.add(_normalize_eq_number(eq))

console.print(f"[cyan]Paper:[/cyan] {doi}")
console.print(f"[cyan]Distinct equations to extract:[/cyan] {sorted(wanted, key=lambda x: (len(x), x))}")

eq_lookup = extract_equations_for_claim(doi, pdf_path, sorted(wanted))

console.rule(f"[bold]Extracted equations ({len(eq_lookup)}/{len(wanted)})[/bold]")
tbl = Table(show_lines=True)
tbl.add_column("Eq #", style="cyan", max_width=6)
tbl.add_column("LaTeX", style="white", max_width=100)
for eq_num in sorted(eq_lookup.keys(), key=lambda x: (len(x), x)):
    e = eq_lookup[eq_num]
    tbl.add_row(e.number, e.latex)
console.print(tbl)

missing = wanted - set(eq_lookup.keys())
if missing:
    console.print(f"\n[yellow]Missing:[/yellow] {sorted(missing)}")

# report cache
cache_dir = get_path("processed_data") / "equations"
pages = list(cache_dir.glob(f"{doi}_p*.json"))
console.print(f"\n[dim]Cache entries for this paper: {len(pages)}[/dim]")
