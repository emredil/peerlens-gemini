import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.claim_extractor import extract_claims, save_claims
from peerlens.settings import get_path
from rich.console import Console
from rich.table import Table
from collections import Counter

console = Console()

doi = "SciPostPhys.21.1.011"
parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"

if not parsed_path.exists():
    src = Path("/Users/emredil/Projects/peerlens/data/processed/parsed") / f"{doi}.json"
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_path.write_text(src.read_text())

parsed = json.loads(parsed_path.read_text())

console.print(f"[cyan]Extracting claims from[/cyan] {doi}")
console.print(f"  title: {parsed['title'][:70]}")
console.print(f"  equations: {parsed['equation_count']}, sections: {len(parsed['sections'])}")

result = extract_claims(parsed)
save_claims(result)

console.rule("[bold]Summary[/bold]")
console.print(f"Total claims:     {result.n_claims_total}")
console.print(f"  dimensional:    {result.n_dimensional}")
console.print(f"  limit:          {result.n_limit}")
console.print(f"  reduction:      {result.n_reduction}")
console.print(f"  algebraic id.:  {result.n_algebraic}")
console.print(f"  symmetry:       {result.n_symmetry}")
console.print(f"  conservation:   {result.n_conservation}")
console.print(f"Symbols catalogued: {len(result.symbols)}")

vm = Counter(c.verification_method for c in result.claims)
console.print(f"Verification methods: {dict(vm)}")
console.print(f"[dim]Model: {result.model}, tokens {result.input_tokens}/{result.output_tokens}[/dim]")

console.rule("[bold]Claims[/bold]")
tbl = Table(show_lines=True)
tbl.add_column("Type", style="cyan", max_width=10)
tbl.add_column("Verify", style="green", max_width=9)
tbl.add_column("Loc", style="yellow", max_width=12)
tbl.add_column("Refs", style="magenta", max_width=10)
tbl.add_column("Claim", style="white", max_width=45)
tbl.add_column("Check hint", style="dim", max_width=35)
for c in result.claims:
    tbl.add_row(c.claim_type, c.verification_method, c.location,
                ", ".join(c.referenced_equations) or "-",
                c.natural_language, c.check_hint)
console.print(tbl)

console.rule("[bold]Symbols[/bold]")
stbl = Table()
stbl.add_column("Symbol", style="cyan")
stbl.add_column("Meaning", style="white", max_width=40)
stbl.add_column("Units", style="yellow", max_width=22)
stbl.add_column("First seen", style="dim")
for s in result.symbols:
    stbl.add_row(s.symbol, s.meaning, s.units, s.first_seen_at)
console.print(stbl)
