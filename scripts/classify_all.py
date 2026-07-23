"""Run the classifier on every parsed paper; save results and evaluation.

Loads ground truth from all SciPostPhys_*.jsonl metadata files, calls the
classifier per paper (idempotent: skips existing results), then computes
per-label precision/recall/F1 and aggregate Jaccard scores.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anthropic import Anthropic
from peerlens.agents.classifier import classify_paper, save_classification
from peerlens.classification.taxonomy import (
    normalize_specialties, normalize_approaches,
    SUBDOMAIN_CODES, PAPER_TYPES,
)
from peerlens.settings import get_path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def load_ground_truth() -> dict[str, dict]:
    """Load all metadata files, return {doi_label: {subdomains, paper_types}}."""
    gt = {}
    meta_dir = get_path("raw_data") / "metadata"
    for f in meta_dir.glob("SciPostPhys*.jsonl"):
        for line in f.open():
            r = json.loads(line)
            gt[r["doi_label"]] = {
                "subdomains": normalize_specialties(r["specialties"]),
                "paper_types": normalize_approaches(r["approaches"]),
                "title": r["title"],
            }
    return gt


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    return p, r, f


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def main(limit: int | None) -> None:
    parsed_dir = get_path("processed_data") / "parsed"
    results_dir = get_path("processed_data") / "classifications"
    results_dir.mkdir(parents=True, exist_ok=True)

    parsed_files = sorted(parsed_dir.glob("SciPostPhys.*.json"))
    if limit:
        parsed_files = parsed_files[:limit]

    gt_map = load_ground_truth()
    console.print(f"[cyan]Corpus:[/cyan] {len(parsed_files)} papers")
    console.print(f"[cyan]Ground truth loaded for:[/cyan] {len(gt_map)} papers")

    client = Anthropic()
    total_cost = 0.0
    total_in = 0
    total_out = 0
    processed = 0
    skipped = 0
    failed = 0
    failures = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Classifying", total=len(parsed_files))
        for pf in parsed_files:
            doi = pf.stem
            out_path = results_dir / f"{doi}.json"
            if out_path.exists():
                skipped += 1
                prog.update(task, advance=1)
                continue
            try:
                parsed = json.loads(pf.read_text())
                result = classify_paper(parsed, client=client)
                save_classification(result, results_dir)
                total_in += result.input_tokens
                total_out += result.output_tokens
                total_cost += (result.input_tokens/1e6)*0.80 + (result.output_tokens/1e6)*4.00
                processed += 1
                time.sleep(0.3)  # be polite to the API
            except Exception as e:
                failed += 1
                failures.append((doi, str(e)))
            prog.update(task, advance=1)

    console.print(f"\n[green]Processed:[/green] {processed}  [yellow]Skipped:[/yellow] {skipped}  [red]Failed:[/red] {failed}")
    console.print(f"[dim]Tokens in/out: {total_in}/{total_out}, cost: ${total_cost:.3f}[/dim]")
    if failures:
        console.print("\n[red]Failures (first 5):[/red]")
        for d, e in failures[:5]:
            console.print(f"  {d}: {e[:100]}")

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    console.rule("[bold cyan]Evaluation[/bold cyan]")

    all_results = {}
    for f in results_dir.glob("SciPostPhys.*.json"):
        d = json.loads(f.read_text())
        all_results[d["doi_label"]] = d

    # per-label counters
    sd_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    pt_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    subdomain_jaccards = []
    paper_type_jaccards = []
    exact_match_sd = 0
    exact_match_pt = 0
    n_evaluated = 0

    for doi, res in all_results.items():
        if doi not in gt_map:
            continue
        n_evaluated += 1
        pred_sd = set(res["subdomains"])
        true_sd = set(gt_map[doi]["subdomains"])
        pred_pt = set(res["paper_types"])
        true_pt = set(gt_map[doi]["paper_types"])

        subdomain_jaccards.append(jaccard(pred_sd, true_sd))
        paper_type_jaccards.append(jaccard(pred_pt, true_pt))
        if pred_sd == true_sd:
            exact_match_sd += 1
        if pred_pt == true_pt:
            exact_match_pt += 1

        for code in SUBDOMAIN_CODES:
            if code in pred_sd and code in true_sd:
                sd_counts[code]["tp"] += 1
            elif code in pred_sd:
                sd_counts[code]["fp"] += 1
            elif code in true_sd:
                sd_counts[code]["fn"] += 1

        for code in PAPER_TYPES:
            if code in pred_pt and code in true_pt:
                pt_counts[code]["tp"] += 1
            elif code in pred_pt:
                pt_counts[code]["fp"] += 1
            elif code in true_pt:
                pt_counts[code]["fn"] += 1

    def avg(xs): return sum(xs)/max(len(xs), 1)

    console.print(f"\n[bold]Papers evaluated:[/bold] {n_evaluated}")
    console.print(f"[bold]Subdomain — mean Jaccard:[/bold] {avg(subdomain_jaccards):.3f}")
    console.print(f"[bold]Subdomain — exact match:[/bold]  {exact_match_sd}/{n_evaluated} ({100*exact_match_sd/max(n_evaluated,1):.1f}%)")
    console.print(f"[bold]Paper type — mean Jaccard:[/bold] {avg(paper_type_jaccards):.3f}")
    console.print(f"[bold]Paper type — exact match:[/bold]  {exact_match_pt}/{n_evaluated} ({100*exact_match_pt/max(n_evaluated,1):.1f}%)")

    console.print("\n[bold]Per-subdomain P/R/F1 (support = ground-truth count):[/bold]")
    sd_table = Table()
    sd_table.add_column("Subdomain", style="cyan")
    sd_table.add_column("TP", style="green", justify="right")
    sd_table.add_column("FP", style="yellow", justify="right")
    sd_table.add_column("FN", style="red", justify="right")
    sd_table.add_column("Precision", justify="right")
    sd_table.add_column("Recall", justify="right")
    sd_table.add_column("F1", style="bold", justify="right")
    sd_table.add_column("Support", justify="right")
    for code in SUBDOMAIN_CODES:
        c = sd_counts[code]
        support = c["tp"] + c["fn"]
        if support == 0 and c["fp"] == 0:
            continue
        p, r, f = prf1(c["tp"], c["fp"], c["fn"])
        sd_table.add_row(code, str(c["tp"]), str(c["fp"]), str(c["fn"]),
                         f"{p:.2f}", f"{r:.2f}", f"{f:.2f}", str(support))
    console.print(sd_table)

    console.print("\n[bold]Per paper-type P/R/F1:[/bold]")
    pt_table = Table()
    pt_table.add_column("Paper type", style="cyan")
    pt_table.add_column("TP", style="green", justify="right")
    pt_table.add_column("FP", style="yellow", justify="right")
    pt_table.add_column("FN", style="red", justify="right")
    pt_table.add_column("Precision", justify="right")
    pt_table.add_column("Recall", justify="right")
    pt_table.add_column("F1", style="bold", justify="right")
    pt_table.add_column("Support", justify="right")
    for code in PAPER_TYPES:
        c = pt_counts[code]
        support = c["tp"] + c["fn"]
        p, r, f = prf1(c["tp"], c["fp"], c["fn"])
        pt_table.add_row(code, str(c["tp"]), str(c["fp"]), str(c["fn"]),
                         f"{p:.2f}", f"{r:.2f}", f"{f:.2f}", str(support))
    console.print(pt_table)

    # Save aggregate report
    report = {
        "n_evaluated": n_evaluated,
        "subdomain_mean_jaccard": avg(subdomain_jaccards),
        "subdomain_exact_match": exact_match_sd,
        "paper_type_mean_jaccard": avg(paper_type_jaccards),
        "paper_type_exact_match": exact_match_pt,
        "subdomain_per_label": {c: dict(sd_counts[c]) for c in SUBDOMAIN_CODES},
        "paper_type_per_label": {c: dict(pt_counts[c]) for c in PAPER_TYPES},
        "total_cost_usd": total_cost,
    }
    report_path = results_dir.parent / "classification_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[dim]Report saved: {report_path}[/dim]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="cap number of papers (for a smaller trial run)")
    args = p.parse_args()
    main(args.limit)
