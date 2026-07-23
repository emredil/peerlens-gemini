"""Batch: extract math claims for the first N theoretical papers.

Selects papers whose SciPost `approaches` include 'theoretical' — these are
the ones where math checking is most valuable. Idempotent.
"""
import argparse
import json
import sys
import time
import ast
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anthropic import Anthropic
from peerlens.agents.claim_extractor import extract_claims, save_claims
from peerlens.settings import get_path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def load_theoretical_dois() -> list[str]:
    """Return DOIs whose approaches include 'theoretical'."""
    meta_dir = get_path("raw_data") / "metadata"
    result = []
    for f in meta_dir.glob("SciPostPhys*.jsonl"):
        for line in f.open():
            r = json.loads(line)
            try:
                apps = ast.literal_eval(r["approaches"])
            except Exception:
                apps = []
            if "theoretical" in apps and r["doi_label"].startswith("SciPostPhys."):
                result.append(r["doi_label"])
    return sorted(set(result))


def main(n: int) -> None:
    parsed_dir = get_path("processed_data") / "parsed"
    claims_dir = get_path("processed_data") / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)

    theo_dois = load_theoretical_dois()
    console.print(f"[cyan]Theoretical papers available:[/cyan] {len(theo_dois)}")
    theo_dois = theo_dois[:n]
    console.print(f"[cyan]Processing:[/cyan] {len(theo_dois)}")

    client = Anthropic()
    total_in = total_out = 0
    total_cost = 0.0
    ok = skip = fail = 0
    failures = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Extracting", total=len(theo_dois))
        for doi in theo_dois:
            out = claims_dir / f"{doi}.json"
            if out.exists():
                skip += 1
                prog.update(task, advance=1)
                continue
            parsed_path = parsed_dir / f"{doi}.json"
            if not parsed_path.exists():
                fail += 1
                failures.append((doi, "parsed not found"))
                prog.update(task, advance=1)
                continue
            try:
                parsed = json.loads(parsed_path.read_text())
                res = extract_claims(parsed, client=client)
                save_claims(res)
                total_in += res.input_tokens
                total_out += res.output_tokens
                total_cost += (res.input_tokens/1e6)*0.80 + (res.output_tokens/1e6)*4.00
                ok += 1
                time.sleep(0.3)
            except Exception as e:
                fail += 1
                failures.append((doi, str(e)))
            prog.update(task, advance=1)

    console.print(f"\n[green]OK:[/green] {ok}  [yellow]Skipped:[/yellow] {skip}  [red]Failed:[/red] {fail}")
    console.print(f"[dim]Tokens: {total_in}/{total_out}, cost: ${total_cost:.3f}[/dim]")
    if failures:
        console.print("\n[red]Failures:[/red]")
        for d, e in failures[:5]:
            console.print(f"  {d}: {e[:100]}")

    # Corpus health aggregation
    console.rule("[bold cyan]Aggregate stats across corpus[/bold cyan]")
    total_claims = 0
    total_symbols = 0
    type_counter = Counter()
    method_counter = Counter()
    n_no_symbols = 0
    n_no_dimensional = 0
    n_files = 0

    for f in claims_dir.glob("SciPostPhys.*.json"):
        d = json.loads(f.read_text())
        n_files += 1
        total_claims += d["n_claims_total"]
        total_symbols += len(d["symbols"])
        if len(d["symbols"]) == 0:
            n_no_symbols += 1
        if d.get("n_dimensional", 0) == 0:
            n_no_dimensional += 1
        for c in d["claims"]:
            type_counter[c["claim_type"]] += 1
            method_counter[c.get("verification_method", "unknown")] += 1

    if n_files > 0:
        console.print(f"Papers with claims: {n_files}")
        console.print(f"  avg claims/paper:   {total_claims/n_files:.1f}")
        console.print(f"  avg symbols/paper:  {total_symbols/n_files:.1f}")
        console.print(f"  papers with 0 symbols:      {n_no_symbols}")
        console.print(f"  papers with 0 dimensional:  {n_no_dimensional}")

        tbl = Table(title="Claim type distribution")
        tbl.add_column("Type", style="cyan")
        tbl.add_column("Count", justify="right")
        tbl.add_column("Share", justify="right")
        for t, c in type_counter.most_common():
            tbl.add_row(t, str(c), f"{100*c/max(total_claims,1):.1f}%")
        console.print(tbl)

        tbl2 = Table(title="Verification method distribution")
        tbl2.add_column("Method", style="cyan")
        tbl2.add_column("Count", justify="right")
        tbl2.add_column("Share", justify="right")
        for t, c in method_counter.most_common():
            tbl2.add_row(t, str(c), f"{100*c/max(total_claims,1):.1f}%")
        console.print(tbl2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()
    main(args.n)
