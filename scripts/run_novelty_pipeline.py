"""Run the novelty agent on a list of papers. Idempotent."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anthropic import Anthropic
from peerlens.agents.novelty_agent import analyze_novelty, save_report
from peerlens.settings import get_path
from rich.console import Console

console = Console()

DEFAULT_DOIS = [
    "SciPostPhys.20.3.080",
    "SciPostPhys.20.4.097",
    "SciPostPhys.21.1.007",
    "SciPostPhys.20.6.166",
    "SciPostPhys.20.4.119",
]


def main(dois: list[str]) -> None:
    client = Anthropic()
    console.rule("[bold cyan]Novelty batch[/bold cyan]")
    console.print(f"Papers: {len(dois)}")

    total_cost = 0.0
    for doi in dois:
        console.rule(f"[bold]{doi}[/bold]")
        out = get_path("processed_data") / "novelty_reports" / f"{doi}.json"
        if out.exists():
            d = json.loads(out.read_text())
            console.print(f"    [dim]cached: {d['n_findings']} findings, "
                          f"{d['n_related_retrieved']} related[/dim]")
            continue
        try:
            rep = analyze_novelty(doi, client=client)
            save_report(rep)
            cost = (rep.input_tokens/1e6)*3.00 + (rep.output_tokens/1e6)*15.00
            total_cost += cost
            console.print(f"    retrieved={rep.n_related_retrieved}, "
                          f"corpus={rep.n_corpus_matches}, scholar={rep.n_scholar_matches}, "
                          f"uncited_close={rep.n_uncited_but_close}")
            console.print(f"    findings={rep.n_findings}  [dim]${cost:.3f}[/dim]")
        except Exception as e:
            console.print(f"    [red]FAILED[/red]: {type(e).__name__}: {str(e)[:200]}")
        time.sleep(1)  # be polite to Semantic Scholar

    console.rule(f"[bold green]Novelty batch complete — total cost ${total_cost:.3f}[/bold green]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--doi", action="append")
    args = p.parse_args()
    dois = args.doi if args.doi else DEFAULT_DOIS
    main(dois)
