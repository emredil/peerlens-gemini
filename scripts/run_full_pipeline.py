"""Run the complete Gemini pipeline on all pipeline papers, plus evaluation."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.gemini_client import GeminiClient
from peerlens.agents.claim_extractor import extract_claims, save_claims
from peerlens.agents.dimensional_checker import check_paper as dim_check, save_report as save_dim
from peerlens.agents.sympy_verifier import verify_paper, save_report as save_sym
from peerlens.agents.novelty_agent import analyze_novelty, save_report as save_nov
from peerlens.agents.structure_agent import analyze_structure, save_report as save_str
from peerlens.orchestrator.math_aggregator import aggregate_and_render
from peerlens.orchestrator.referee_aggregator import aggregate_and_render_full
from peerlens.evaluation.referee_comparison import compare, save_comparison
from peerlens.settings import get_path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_DOIS = [
    "SciPostPhys.20.3.080",
    "SciPostPhys.20.4.097",
    "SciPostPhys.21.1.007",
    "SciPostPhys.20.6.166",
    "SciPostPhys.20.4.119",
    "SciPostPhys.21.1.011",
]


def ensure_parsed(doi: str) -> Path:
    dst = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not dst.exists():
        src = Path("/Users/emredil/Projects/peerlens/data/processed/parsed") / f"{doi}.json"
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    return dst


def stage_claims(doi: str, client: GeminiClient) -> dict:
    out = get_path("processed_data") / "claims" / f"{doi}.json"
    if out.exists():
        console.print(f"    [dim]claims: cached[/dim]")
        return json.loads(out.read_text())
    parsed_path = ensure_parsed(doi)
    parsed = json.loads(parsed_path.read_text())
    res = extract_claims(parsed, client=client)
    save_claims(res)
    console.print(f"    claims: [green]{res.n_claims_total}[/green] "
                  f"(sympy={sum(1 for c in res.claims if c.verification_method=='sympy')}) "
                  f"[dim]tokens {res.input_tokens}/{res.output_tokens}[/dim]")
    return json.loads(out.read_text())


def stage_dimensional(doi: str, claims_data: dict, client: GeminiClient) -> None:
    out = get_path("processed_data") / "dimensional_reports" / f"{doi}.json"
    if out.exists():
        console.print(f"    [dim]dimensional: cached[/dim]")
        return
    pdf = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    rep = dim_check(claims_data, pdf, max_vision_calls=2, client=client)
    save_dim(rep)
    console.print(f"    dimensional: [green]{rep.n_findings}[/green] "
                  f"(crit={rep.n_critical}, warn={rep.n_warning}, info={rep.n_info})")


def stage_sympy(doi: str, claims_data: dict, client: GeminiClient) -> None:
    out = get_path("processed_data") / "sympy_reports" / f"{doi}.json"
    if out.exists():
        console.print(f"    [dim]sympy: cached[/dim]")
        return
    pdf = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    rep = verify_paper(claims_data, pdf, max_claims=8, client=client)
    save_sym(rep)
    console.print(f"    sympy: verified={rep.n_verified}, refuted={rep.n_refuted}, "
                  f"unable={rep.n_unable}")


def stage_novelty(doi: str, client: GeminiClient) -> None:
    out = get_path("processed_data") / "novelty_reports" / f"{doi}.json"
    if out.exists():
        console.print(f"    [dim]novelty: cached[/dim]")
        return
    rep = analyze_novelty(doi, client=client)
    save_nov(rep)
    console.print(f"    novelty: retrieved={rep.n_related_retrieved} findings={rep.n_findings}")


def stage_structure(doi: str, client: GeminiClient) -> None:
    out = get_path("processed_data") / "structure_reports" / f"{doi}.json"
    if out.exists():
        console.print(f"    [dim]structure: cached[/dim]")
        return
    rep = analyze_structure(doi, client=client)
    save_str(rep)
    console.print(f"    structure: candidates={rep.n_candidates_scanned} findings={rep.n_findings} "
                  f"(warn={rep.n_warning})")


def stage_aggregate(doi: str) -> None:
    aggregate_and_render(doi)
    aggregate_and_render_full(doi)


def stage_compare(doi: str, client: GeminiClient) -> dict:
    out = get_path("processed_data") / "evaluations" / f"{doi}_comparison.json"
    if out.exists():
        return json.loads(out.read_text())
    res = compare(doi, client=client)
    save_comparison(res)
    return json.loads(out.read_text())


def main(dois: list[str]) -> None:
    client = GeminiClient()
    console.rule("[bold cyan]Gemini full pipeline batch[/bold cyan]")
    console.print(f"Papers: {len(dois)}")

    for doi in dois:
        console.rule(f"[bold]{doi}[/bold]")
        try:
            claims_data = stage_claims(doi, client)
            stage_dimensional(doi, claims_data, client)
            stage_sympy(doi, claims_data, client)
            stage_novelty(doi, client)
            stage_structure(doi, client)
            stage_aggregate(doi)
        except Exception as e:
            console.print(f"    [red]FAILED[/red]: {type(e).__name__}: {str(e)[:250]}")
        time.sleep(1)

    console.rule("[bold cyan]Evaluation batch[/bold cyan]")
    results = []
    for doi in dois:
        try:
            r = stage_compare(doi, client)
            results.append((doi, r))
            console.print(f"  {doi}: ai={r['n_ai_findings']} "
                          f"ref={r['n_referee_math_concerns']} "
                          f"match={r['n_matches']} "
                          f"missed={r['n_missed']} "
                          f"novel={r['n_novel']}")
        except Exception as e:
            console.print(f"  {doi}: [red]comparison FAILED[/red]: {type(e).__name__}: {str(e)[:200]}")
        time.sleep(1)

    console.rule("[bold cyan]Dashboard[/bold cyan]")
    tbl = Table(show_lines=True)
    tbl.add_column("DOI", style="cyan", max_width=25)
    tbl.add_column("AI", justify="right")
    tbl.add_column("Ref", justify="right")
    tbl.add_column("Match", justify="right", style="green")
    tbl.add_column("Miss", justify="right", style="red")
    tbl.add_column("Novel", justify="right", style="yellow")
    tbl.add_column("Prec", justify="right")
    tbl.add_column("Rec", justify="right")

    tot_ai = tot_ref = tot_match = tot_miss = tot_nov = 0
    for doi, r in results:
        ai = r['n_ai_findings']
        ref = r['n_referee_math_concerns']
        m = r['n_matches']
        ms = r['n_missed']
        n = r['n_novel']
        prec = m / ai if ai else 0.0
        rec = m / ref if ref else 0.0
        tbl.add_row(doi, str(ai), str(ref), str(m), str(ms), str(n),
                    f"{prec:.0%}", f"{rec:.0%}")
        tot_ai += ai; tot_ref += ref; tot_match += m; tot_miss += ms; tot_nov += n

    tot_prec = tot_match / tot_ai if tot_ai else 0.0
    tot_rec = tot_match / tot_ref if tot_ref else 0.0
    tbl.add_row("[bold]TOTAL[/bold]", f"[bold]{tot_ai}[/bold]", f"[bold]{tot_ref}[/bold]",
                f"[bold]{tot_match}[/bold]", f"[bold]{tot_miss}[/bold]", f"[bold]{tot_nov}[/bold]",
                f"[bold]{tot_prec:.0%}[/bold]", f"[bold]{tot_rec:.0%}[/bold]")

    console.print(tbl)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--doi", action="append")
    args = p.parse_args()
    dois = args.doi if args.doi else DEFAULT_DOIS
    main(dois)
