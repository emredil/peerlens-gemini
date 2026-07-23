"""Fetch and parse referee reports for the 3 test publications."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.ingestion.scipost import (
    PublicationRecord,
    fetch_reports_for_publication,
    save_reports,
)
from peerlens.settings import get_path
from rich.console import Console
from rich.table import Table

console = Console()

# Load metadata saved earlier
metadata_path = get_path("raw_data") / "metadata" / "SciPostPhys_first_3.jsonl"
records = []
with metadata_path.open() as f:
    for line in f:
        records.append(PublicationRecord(**json.loads(line)))

console.print(f"[cyan]Loaded[/cyan] {len(records)} records")

for r in records:
    console.print(f"\n[bold]Fetching reports for {r.doi_label}[/bold]")
    data = fetch_reports_for_publication(r)
    path = save_reports(r, data)

    table = Table(title=f"Reports for {r.doi_label}", show_lines=True)
    table.add_column("V", style="cyan")
    table.add_column("#", style="yellow")
    table.add_column("Identity", style="magenta", max_width=18)
    table.add_column("Date", style="green")
    table.add_column("Recommendation", style="white", max_width=22)
    table.add_column("Scores (v/s/o/c/f/g)", style="blue", max_width=22)
    table.add_column("Body", style="dim")
    table.add_column("Reply?", style="red")

    for version, reports in sorted(data["versions"].items()):
        for rep in reports:
            scores_short = "/".join(
                (v or "?")[:4] for v in [
                    rep.scores.get("validity"),
                    rep.scores.get("significance"),
                    rep.scores.get("originality"),
                    rep.scores.get("clarity"),
                    rep.scores.get("formatting"),
                    rep.scores.get("grammar"),
                ]
            )
            table.add_row(
                str(version),
                str(rep.report_number),
                rep.referee_identity[:18],
                rep.date,
                (rep.recommendation or "-")[:22],
                scores_short,
                f"{len(rep.body)}c",
                "Y" if rep.author_reply else "N",
            )
    console.print(table)
    console.print(f"[dim]Saved: {path}[/dim]")

# Quick sanity check on one JSON
console.print("\n[bold cyan]JSON sanity check on SciPostPhys.21.1.011:[/bold cyan]")
with (get_path("raw_data") / "reports" / "SciPostPhys.21.1.011.json").open() as f:
    d = json.load(f)
first_report = d["versions"]["1"][0]  # first report of v1
console.print(f"  scores: {first_report['scores']}")
console.print(f"  strengths (first 100 chars): {first_report['strengths'][:100]}")
console.print(f"  body ends with: ...{first_report['body'][-80:]}")
console.print(f"  has author reply: {first_report['author_reply'] is not None}")
if first_report['author_reply']:
    console.print(f"  reply from: {first_report['author_reply']['author_name']}")
