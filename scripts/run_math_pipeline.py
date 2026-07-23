"""Run the full Phase 1a pipeline on a list of papers.

For each paper: claim extraction → dimensional check → sympy verification → aggregation.
Skips stages whose output already exists on disk (idempotent).

Does NOT run the referee comparison (that is a separate script; the
comparison uses Sonnet-5 and we want to review the math reports first).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anthropic import Anthropic
from peerlens.agents.claim_extractor import extract_claims, save_claims
from peerlens.agents.dimensional_checker import check_paper as dimensional_check, save_report as save_dim_report
from peerlens.agents.sympy_verifier import verify_paper, save_report as save_sympy_report
from peerlens.orchestrator.math_aggregator import aggregate_and_render
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


def stage_claims(doi: str, client: Anthropic) -> dict:
    out = get_path("processed_data") / "claims" / f"{doi}.json"
    if out.exists():
        console.print(f"    [dim]claims: cached[/dim]")
        return json.loads(out.read_text())
    parsed = json.loads((get_path("processed_data") / "parsed" / f"{doi}.json").read_text())
    res = extract_claims(parsed, client=client)
    save_claims(res)
    console.print(f"    claims: [green]{res.n_claims_total}[/green] "
                  f"(sympy={sum(1 for c in res.claims if c.verification_method=='sympy')}) "
                  f"[dim]tokens {res.input_tokens}/{res.output_tokens}[/dim]")
    return json.loads(out.read_text())


def stage_dimensional(doi: str, claims_data: dict, client: Anthropic):
    out = get_path("processed_data") / "dimensional_reports" / f"{doi}.json"
    if out.exists():
        console.print(f"    [dim]dimensional: cached[/dim]")
        return
    pdf = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    rep = dimensional_check(claims_data, pdf, max_vision_calls=2, client=client)
    save_dim_report(rep)
    console.print(f"    dimensional: [green]{rep.n_findings}[/green] "
                  f"(crit={rep.n_critical}, warn={rep.n_warning}, info={rep.n_info})")


def stage_sympy(doi: str, claims_data: dict, client: Anthropic):
    out = get_path("processed_data") / "sympy_reports" / f"{doi}.json"
    if out.exists():
        console.print(f"    [dim]sympy: cached[/dim]")
        return
    pdf = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    rep = verify_paper(claims_data, pdf, max_claims=8, client=client)
    save_sympy_report(rep)
    console.print(f"    sympy: verified={rep.n_verified}, refuted={rep.n_refuted}, "
                  f"unable={rep.n_unable}")


def stage_aggregate(doi: str):
    json_path, md_path = aggregate_and_render(doi)
    console.print(f"    [green]aggregated →[/green] {md_path.name}")


def main(dois: list[str]) -> None:
    client = Anthropic()
    console.rule("[bold cyan]Math pipeline batch[/bold cyan]")
    console.print(f"Papers: {len(dois)}")

    for doi in dois:
        console.rule(f"[bold]{doi}[/bold]")
        try:
            claims_data = stage_claims(doi, client)
            stage_dimensional(doi, claims_data, client)
            stage_sympy(doi, claims_data, client)
            stage_aggregate(doi)
        except Exception as e:
            console.print(f"    [red]FAILED[/red]: {type(e).__name__}: {str(e)[:200]}")
        time.sleep(1)

    console.rule("[bold green]Batch complete[/bold green]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--doi", action="append", help="specific DOI(s); defaults to preset list")
    args = p.parse_args()
    dois = args.doi if args.doi else DEFAULT_DOIS
    main(dois)
