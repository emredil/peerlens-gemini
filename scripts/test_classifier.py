import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.classifier import classify_paper
from peerlens.classification.taxonomy import normalize_specialties, normalize_approaches
from peerlens.settings import get_path
from rich.console import Console

console = Console()

doi = "SciPostPhys.21.1.011"
parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"

if not parsed_path.exists():
    console.print(f"[yellow]parsed JSON not found at {parsed_path}. Copying from peerlens original...[/yellow]")
    src = Path("/Users/emredil/Projects/peerlens/data/processed/parsed") / f"{doi}.json"
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_path.write_text(src.read_text())

parsed = json.loads(parsed_path.read_text())

meta_dir = get_path("raw_data") / "metadata"
gt = None
for f in meta_dir.glob("SciPostPhys*.jsonl"):
    for line in f.open():
        r = json.loads(line)
        if r["doi_label"] == doi:
            gt = r
            break
    if gt: break

gt_sd = normalize_specialties(gt["specialties"]) if gt else []
gt_pt = normalize_approaches(gt["approaches"]) if gt else []

console.print(f"[cyan]Classifying[/cyan] {doi}")
console.print(f"  title: {parsed['title'][:80]}")

result = classify_paper(parsed)

console.rule("[bold]Prediction (Gemini)[/bold]")
console.print(f"Subdomains:  {result.subdomains}")
console.print(f"Paper types: {result.paper_types}")
console.print(f"Rationale:   {result.rationale}")
console.print(f"[dim]Model: {result.model}, tokens {result.input_tokens}/{result.output_tokens}[/dim]")

console.rule("[bold]Ground truth (SciPost)[/bold]")
console.print(f"Subdomains:  {gt_sd}")
console.print(f"Paper types: {gt_pt}")

sd_pred, sd_true = set(result.subdomains), set(gt_sd)
pt_pred, pt_true = set(result.paper_types), set(gt_pt)

def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)

console.rule("[bold]Agreement[/bold]")
console.print(f"Subdomain Jaccard: {jaccard(sd_pred, sd_true):.2f}")
console.print(f"Paper type Jaccard: {jaccard(pt_pred, pt_true):.2f}")
