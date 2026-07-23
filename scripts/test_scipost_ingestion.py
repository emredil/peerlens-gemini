"""Smoke test for SciPost ingestion.

Fetches 3 records, downloads their PDFs, and prints a small report.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.ingestion.scipost import fetch_batch
from rich.console import Console
from rich.table import Table

console = Console()

records = fetch_batch(journal="SciPostPhys", n=3, download_pdfs=True)

table = Table(title="Fetched Publications")
table.add_column("DOI Label", style="cyan", no_wrap=True)
table.add_column("Title", style="white", max_width=50)
table.add_column("Specialties", style="green")
table.add_column("Approaches", style="yellow")
table.add_column("Sub URL?", style="magenta")

for r in records:
    table.add_row(
        r.doi_label,
        r.title[:50] + ("..." if len(r.title) > 50 else ""),
        ", ".join(r.specialties),
        r.approaches,
        "yes" if r.submission_url else "no",
    )

console.print(table)

# Check PDFs on disk
from peerlens.settings import get_path
pdf_dir = get_path("raw_data") / "pdf"
pdfs = sorted(pdf_dir.glob("*.pdf"))
console.print(f"\n[green]PDFs on disk:[/green] {len(pdfs)}")
for p in pdfs:
    console.print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
