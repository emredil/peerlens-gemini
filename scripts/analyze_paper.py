"""One-shot pipeline for a single paper: PDF -> full AI Referee Section.

Usage:
    python scripts/analyze_paper.py --doi SciPostPhys.21.1.011
    python scripts/analyze_paper.py --pdf /path/to/paper.pdf --doi custom_id

Assumes the parsed JSON is already produced. If a raw PDF is given without a
parsed JSON, the ingestion step (title/abstract/sections/references) will
run first.
"""
import argparse
import json
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
from peerlens.ingestion.pdf import parse_pdf, save_parsed
from peerlens.settings import get_path
from rich.console import Console

console = Console()


def ensure_parsed(doi: str, pdf_path: Path | None) -> dict:
    """Return parsed dict for this DOI, running ingestion if needed."""
    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if parsed_path.exists():
        return json.loads(parsed_path.read_text())

    if pdf_path is None:
        pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"No parsed JSON and no PDF at {pdf_path}")

    console.print(f"  [dim]parsing PDF: {pdf_path.name}[/dim]")
    parsed = parse_pdf(pdf_path, doi_label=doi)
    save_parsed(parsed)
    return parsed


def run_pipeline(doi: str, pdf_path: Path | None = None) -> None:
    client = GeminiClient()
    total_t0 = time.time()

    console.rule(f"[bold cyan]PeerLens pipeline — {doi}[/bold cyan]")

    parsed = ensure_parsed(doi, pdf_path)
    resolved_pdf = pdf_path or (get_path("raw_data") / "pdf" / f"{doi}.pdf")

    console.print("[cyan]1/7[/cyan] Claim extraction")
    claims_path = get_path("processed_data") / "claims" / f"{doi}.json"
    if claims_path.exists():
        console.print("       [dim]cached[/dim]")
        claims_data = json.loads(claims_path.read_text())
    else:
        res = extract_claims(parsed, client=client)
        save_claims(res)
        console.print(f"       {res.n_claims_total} claims, {len(res.symbols)} symbols")
        claims_data = json.loads(claims_path.read_text())

    console.print("[cyan]2/7[/cyan] Dimensional analysis")
    dim_path = get_path("processed_data") / "dimensional_reports" / f"{doi}.json"
    if dim_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        rep = dim_check(claims_data, resolved_pdf, max_vision_calls=2, client=client)
        save_dim(rep)
        console.print(f"       {rep.n_findings} findings "
                      f"(crit={rep.n_critical}, warn={rep.n_warning})")

    console.print("[cyan]3/7[/cyan] Symbolic verification (may fetch equation LaTeX)")
    sym_path = get_path("processed_data") / "sympy_reports" / f"{doi}.json"
    if sym_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        rep = verify_paper(claims_data, resolved_pdf, max_claims=8, client=client)
        save_sym(rep)
        console.print(f"       verified={rep.n_verified}, refuted={rep.n_refuted}, "
                      f"unable={rep.n_unable}")

    console.print("[cyan]4/7[/cyan] Novelty / literature review")
    nov_path = get_path("processed_data") / "novelty_reports" / f"{doi}.json"
    if nov_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        rep = analyze_novelty(doi, client=client)
        save_nov(rep)
        console.print(f"       {rep.n_related_retrieved} related retrieved, "
                      f"{rep.n_findings} findings")

    console.print("[cyan]5/7[/cyan] Structure & narrative")
    str_path = get_path("processed_data") / "structure_reports" / f"{doi}.json"
    if str_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        rep = analyze_structure(doi, client=client)
        save_str(rep)
        console.print(f"       {rep.n_candidates_scanned} candidates scanned, "
                      f"{rep.n_findings} findings")

    console.print("[cyan]6/7[/cyan] Math aggregation")
    aggregate_and_render(doi)

    console.print("[cyan]7/7[/cyan] Full referee section")
    json_path, md_path = aggregate_and_render_full(doi)

    elapsed = time.time() - total_t0
    console.rule("[bold green]Done[/bold green]")
    console.print(f"Total time:      {elapsed:.1f}s")
    console.print(f"Markdown report: {md_path}")
    console.print(f"JSON dump:       {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--doi", required=True, help="paper identifier (used as filename stem)")
    p.add_argument("--pdf", type=Path, default=None, help="path to PDF if not in data/raw/pdf/{doi}.pdf")
    args = p.parse_args()
    run_pipeline(args.doi, args.pdf)
