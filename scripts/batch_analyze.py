"""Batch pipeline runner for PeerLens editorial triage.

Usage:
    python scripts/batch_analyze.py --doi-list list.txt
    python scripts/batch_analyze.py --pdf-dir /path/to/pdfs/
    python scripts/batch_analyze.py --doi SciPostPhys.21.1.011 --doi SciPostPhys.20.4.097

Every paper is run through the full 5-agent pipeline. Failures for
individual papers are logged and skipped; the batch continues.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.gemini_client import GeminiClient
from peerlens.agents.claim_extractor import extract_claims, save_claims
from peerlens.agents.dimensional_checker import check_paper as dim_check, save_report as save_dim
from peerlens.agents.sympy_verifier import verify_paper, save_report as save_sym
from peerlens.agents.novelty_agent import analyze_novelty, save_report as save_nov
from peerlens.agents.structure_agent import analyze_structure, save_report as save_str
from peerlens.agents.domain_expert import analyze_domain, save_report as save_dom
from peerlens.agents.reproducibility_agent import analyze_reproducibility, save_report as save_repro
from peerlens.agents.llm_detector import analyze_llm_authorship, save_report as save_llm
from peerlens.orchestrator.math_aggregator import aggregate_and_render
from peerlens.orchestrator.referee_aggregator import aggregate_and_render_full
from peerlens.ingestion.pdf import parse_pdf, save_parsed
from peerlens.settings import get_path
from rich.console import Console

console = Console()


def resolve_pdf_for_doi(doi: str, pdf_override: Path | None = None) -> Path:
    if pdf_override is not None:
        return pdf_override
    return get_path("raw_data") / "pdf" / f"{doi}.pdf"


def ensure_parsed(doi: str, pdf_path: Path) -> dict:
    """Return the parsed paper as a dict. Runs ingestion if needed and
    reads back the JSON that save_parsed writes."""
    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if parsed_path.exists():
        return json.loads(parsed_path.read_text())
    parsed_obj = parse_pdf(pdf_path, doi_label=doi)
    save_parsed(parsed_obj)
    if not parsed_path.exists():
        raise RuntimeError(
            f"parse_pdf ran for {doi} but save_parsed did not produce "
            f"{parsed_path}."
        )
    return json.loads(parsed_path.read_text())


def run_one(doi: str, pdf_path: Path, client: GeminiClient, *, check_llm: bool = False) -> dict:
    """Return a summary dict with score / verdict / status."""
    t0 = time.time()
    stages_done = []
    try:
        parsed = ensure_parsed(doi, pdf_path)
        stages_done.append("parse")

        claims_path = get_path("processed_data") / "claims" / f"{doi}.json"
        if not claims_path.exists():
            res = extract_claims(parsed, client=client)
            save_claims(res)
        claims_data = json.loads(claims_path.read_text())
        stages_done.append("claims")

        dim_path = get_path("processed_data") / "dimensional_reports" / f"{doi}.json"
        if not dim_path.exists():
            rep = dim_check(claims_data, pdf_path, max_vision_calls=2, client=client)
            save_dim(rep)
        stages_done.append("dimensional")

        sym_path = get_path("processed_data") / "sympy_reports" / f"{doi}.json"
        if not sym_path.exists():
            rep = verify_paper(claims_data, pdf_path, max_claims=8, client=client)
            save_sym(rep)
        stages_done.append("sympy")

        nov_path = get_path("processed_data") / "novelty_reports" / f"{doi}.json"
        if not nov_path.exists():
            rep = analyze_novelty(doi, client=client)
            save_nov(rep)
        stages_done.append("novelty")

        str_path = get_path("processed_data") / "structure_reports" / f"{doi}.json"
        if not str_path.exists():
            rep = analyze_structure(doi, client=client)
            save_str(rep)
        stages_done.append("structure")

        dom_path = get_path("processed_data") / "domain_reports" / f"{doi}.json"
        if not dom_path.exists():
            rep = analyze_domain(doi, client=client)
            save_dom(rep)
        stages_done.append("domain")

        repro_path = get_path("processed_data") / "reproducibility_reports" / f"{doi}.json"
        if not repro_path.exists():
            rep = analyze_reproducibility(doi, client=client)
            save_repro(rep)
        stages_done.append("reproducibility")

        aggregate_and_render(doi)
        stages_done.append("math_aggregate")

        aggregate_and_render_full(doi)
        stages_done.append("referee_aggregate")

        llm_verdict = None
        llm_confidence = None
        if check_llm:
            llm_path = get_path("processed_data") / "llm_detection_reports" / f"{doi}.json"
            if not llm_path.exists():
                rep = analyze_llm_authorship(doi, client=client)
                save_llm(rep)
            if llm_path.exists():
                ld = json.loads(llm_path.read_text())
                llm_verdict = ld.get("verdict")
                llm_confidence = ld.get("confidence")
                # Re-render markdown to embed the Policy section.
                aggregate_and_render_full(doi)
            stages_done.append("llm_detection")

        triage_path = get_path("processed_data") / "triage_scores" / f"{doi}.json"
        triage = json.loads(triage_path.read_text())
        return {
            "doi": doi,
            "status": "ok",
            "score": triage.get("score", 0),
            "verdict": triage.get("verdict", "unknown"),
            "llm_verdict": llm_verdict,
            "llm_confidence": llm_confidence,
            "elapsed_s": round(time.time() - t0, 1),
            "stages_done": stages_done,
        }
    except Exception as e:
        return {
            "doi": doi,
            "status": "failed",
            "error_type": type(e).__name__,
            "error_msg": str(e)[:300],
            "elapsed_s": round(time.time() - t0, 1),
            "stages_done": stages_done,
            "traceback_last": traceback.format_exc().splitlines()[-1],
        }


def load_dois_from_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def load_dois_from_pdf_dir(pdf_dir: Path) -> list[tuple[str, Path]]:
    """Return (doi_label, pdf_path) pairs, using filename stem as doi_label."""
    out = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        out.append((pdf.stem, pdf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi", action="append", help="Explicit DOI (repeatable)")
    ap.add_argument("--doi-list", type=Path, help="File with one DOI per line")
    ap.add_argument("--pdf-dir", type=Path, help="Process every PDF in this directory")
    ap.add_argument("--check-llm", action="store_true",
                    help="Also run LLM-authorship detection on each paper (opt-in).")
    ap.add_argument("--summary-out", type=Path,
                    default=Path("data/processed/batch_summary.json"),
                    help="Where to write the per-paper summary")
    args = ap.parse_args()

    jobs: list[tuple[str, Path | None]] = []
    if args.doi:
        for d in args.doi:
            jobs.append((d, None))
    if args.doi_list:
        for d in load_dois_from_list(args.doi_list):
            jobs.append((d, None))
    if args.pdf_dir:
        for doi, pdf in load_dois_from_pdf_dir(args.pdf_dir):
            jobs.append((doi, pdf))

    if not jobs:
        console.print("[red]No papers to process. Pass --doi, --doi-list, or --pdf-dir.[/red]")
        sys.exit(2)

    console.rule(f"[bold cyan]PeerLens batch — {len(jobs)} paper(s)[/bold cyan]")
    client = GeminiClient()

    summaries: list[dict] = []
    for i, (doi, pdf_override) in enumerate(jobs, 1):
        pdf_path = resolve_pdf_for_doi(doi, pdf_override)
        console.rule(f"[bold]{i}/{len(jobs)}  {doi}[/bold]")
        if not pdf_path.exists():
            console.print(f"  [red]PDF missing at {pdf_path}, skipping[/red]")
            summaries.append({"doi": doi, "status": "no_pdf", "score": None, "verdict": None})
            continue
        summary = run_one(doi, pdf_path, client, check_llm=args.check_llm)
        summaries.append(summary)
        if summary["status"] == "ok":
            v = summary["verdict"]
            color = {"priority": "green", "standard": "yellow", "reject": "red"}.get(v, "white")
            console.print(f"  → [{color}]{v.upper()}[/{color}] "
                          f"(score {summary['score']}/100)  {summary['elapsed_s']}s")
        else:
            console.print(f"  → [red]FAILED[/red] {summary.get('error_type','?')}: "
                          f"{summary.get('error_msg','')[:120]}")

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    console.rule("[bold cyan]Batch complete[/bold cyan]")
    console.print(f"Summary written to: {args.summary_out}")

    n_ok = sum(1 for s in summaries if s.get("status") == "ok")
    n_priority = sum(1 for s in summaries if s.get("verdict") == "priority")
    n_standard = sum(1 for s in summaries if s.get("verdict") == "standard")
    n_reject = sum(1 for s in summaries if s.get("verdict") == "reject")
    n_failed = sum(1 for s in summaries if s.get("status") == "failed")
    console.print(f"Processed: {n_ok}  |  priority: {n_priority}  "
                  f"standard: {n_standard}  reject: {n_reject}  failed: {n_failed}")

    # Refresh the editorial dashboard via subprocess (isolates import scope).
    import subprocess
    dash_script = Path(__file__).parent / "build_editorial_dashboard.py"
    try:
        subprocess.run([sys.executable, str(dash_script)], check=True,
                       capture_output=True, text=True, timeout=30)
        dashboard_path = get_path("processed_data") / "editorial_dashboard.html"
        console.print(f"Dashboard refreshed: {dashboard_path}")
    except subprocess.CalledProcessError as e:
        console.print(f"[yellow]Dashboard refresh failed: {e.stderr[:200]}[/yellow]")


if __name__ == "__main__":
    main()
