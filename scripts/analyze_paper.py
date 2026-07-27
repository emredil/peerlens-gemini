"""End-to-end PeerLens pipeline for a single paper.

Inputs (pick one):
    --pdf /path/to/manuscript.pdf   # local PDF
    --doi SciPostPhys.21.1.011      # existing paper already in data/raw/pdf/
    --arxiv 2401.12345              # arXiv id or URL (downloads PDF)

Behaviour:
  - --pdf: file is copied into data/raw/pdf/ using its filename stem
    as doi_label. Pass --doi to override the stem.
  - --arxiv: PDF is fetched from arXiv and given a doi_label of
    arxiv_{id}. Metadata (title, authors) is printed for confirmation.
  - --doi (no --pdf): PDF must already exist at data/raw/pdf/{doi}.pdf.

Every stage is idempotent: rerunning skips work whose output already
exists. The editorial dashboard is refreshed at the end.
"""
import argparse
import json
import shutil
import subprocess
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
from peerlens.agents.domain_expert import analyze_domain, save_report as save_dom
from peerlens.agents.reproducibility_agent import analyze_reproducibility, save_report as save_repro
from peerlens.agents.llm_detector import analyze_llm_authorship, save_report as save_llm
from peerlens.orchestrator.math_aggregator import aggregate_and_render
from peerlens.orchestrator.referee_aggregator import aggregate_and_render_full
from peerlens.ingestion.pdf import parse_pdf, save_parsed
from peerlens.ingestion.arxiv import (
    fetch_metadata as arxiv_fetch_metadata,
    download_pdf as arxiv_download_pdf,
    arxiv_id_to_doi_label,
)
from peerlens.ingestion.url_fetch import (
    download_pdf as url_download_pdf,
    label_from_url,
)
from peerlens.agents.classifier import classify_paper, save_classification
from peerlens.settings import get_path
from rich.console import Console

console = Console()


def _sanitize_doi_label(raw: str) -> str:
    """Turn a filename stem into a safe identifier for the data tree."""
    cleaned = "".join(c if c.isalnum() or c in ".-_" else "_" for c in raw)
    return cleaned or "manuscript"


def resolve_input(args) -> tuple[str, Path]:
    """Return (doi_label, pdf_path). Copies external PDFs or downloads from arXiv."""
    pdf_dir = get_path("raw_data") / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "url", None):
        url = args.url.strip()
        doi_label = label_from_url(url)
        console.print(f"  Source URL:  [white]{url}[/white]")
        console.print(f"  Label:       {doi_label}")
        pdf_path = pdf_dir / f"{doi_label}.pdf"
        if not pdf_path.exists():
            console.print(f"  [dim]downloading PDF to {pdf_path.name}...[/dim]")
            url_download_pdf(url, pdf_path)
        return doi_label, pdf_path

    if args.arxiv:
        doi_label = arxiv_id_to_doi_label(args.arxiv)
        pdf_path = pdf_dir / f"{doi_label}.pdf"
        if not pdf_path.exists():
            console.print(f"  [dim]fetching arXiv metadata for {args.arxiv}...[/dim]")
            try:
                meta = arxiv_fetch_metadata(args.arxiv)
                if meta:
                    console.print(f"  Title:   [white]{meta.title[:100]}[/white]")
                    if meta.authors:
                        console.print(f"  Authors: {', '.join(meta.authors[:4])}"
                                      + (f" +{len(meta.authors)-4} more" if len(meta.authors) > 4 else ""))
                    if meta.doi:
                        console.print(f"  DOI:     {meta.doi}")
                    if meta.primary_category:
                        console.print(f"  Cat:     {meta.primary_category}")
            except Exception as e:
                console.print(f"  [yellow]metadata fetch failed: {e}[/yellow]")
            console.print(f"  [dim]downloading PDF to {pdf_path.name}...[/dim]")
            arxiv_download_pdf(args.arxiv, pdf_path)
        return doi_label, pdf_path

    if args.pdf:
        src = Path(args.pdf).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"PDF not found: {src}")
        doi_label = _sanitize_doi_label(args.doi or src.stem)
        dst = pdf_dir / f"{doi_label}.pdf"
        if dst.resolve() != src:
            if not dst.exists():
                shutil.copy(src, dst)
                console.print(f"  [dim]copied {src.name} -> {dst}[/dim]")
        return doi_label, dst

    if args.doi:
        pdf_path = pdf_dir / f"{args.doi}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"No PDF at {pdf_path}. Pass --pdf or --arxiv for a "
                f"manuscript we haven't ingested yet."
            )
        return args.doi, pdf_path

    raise SystemExit("Provide --pdf, --doi, or --arxiv.")


def ensure_parsed(doi: str, pdf_path: Path) -> dict:
    """Return a dict of the parsed paper, running ingestion if needed."""
    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if parsed_path.exists():
        return json.loads(parsed_path.read_text())

    parsed_obj = parse_pdf(pdf_path, doi_label=doi)
    save_parsed(parsed_obj)

    if not parsed_path.exists():
        raise RuntimeError(
            f"parse_pdf ran for {doi} but save_parsed did not produce "
            f"{parsed_path}. Check the ingestion module contract."
        )
    return json.loads(parsed_path.read_text())


def _refresh_dashboard() -> None:
    """Rebuild the HTML dashboard as a subprocess (isolates import scope)."""
    script = Path(__file__).parent / "build_editorial_dashboard.py"
    try:
        subprocess.run([sys.executable, str(script)], check=True,
                       capture_output=True, text=True, timeout=30)
    except subprocess.CalledProcessError as e:
        console.print(f"  [yellow]dashboard refresh failed: {e.stderr[:200]}[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]dashboard refresh error: {e}[/yellow]")


def run_pipeline(doi: str, pdf_path: Path, *, check_llm: bool = False) -> None:
    client = GeminiClient()
    total_t0 = time.time()

    console.rule(f"[bold cyan]PeerLens — {doi}[/bold cyan]")

    parsed = ensure_parsed(doi, pdf_path)

    console.print("[cyan]1/10[/cyan] Classification (subdomain + type + field)")
    cls_path = get_path("processed_data") / "classifications" / f"{doi}.json"
    if cls_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        cls = classify_paper(parsed, client=client)
        save_classification(cls, cls_path.parent)
        console.print(f"       subdomains={cls.subdomains}, field={cls.field_top}, types={cls.paper_types}")
    cls_data = json.loads(cls_path.read_text())
    field_top = cls_data.get("field_top", "physics")

    console.print("[cyan]2/10[/cyan] Claim extraction (field={0})".format(field_top))
    claims_path = get_path("processed_data") / "claims" / f"{doi}.json"
    if claims_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        res = extract_claims(parsed, client=client, field=field_top)
        save_claims(res)
        console.print(f"       {res.n_claims_total} claims, {len(res.symbols)} symbols")
    claims_data = json.loads(claims_path.read_text())

    console.print("[cyan]3/10[/cyan] Dimensional analysis")
    dim_path = get_path("processed_data") / "dimensional_reports" / f"{doi}.json"
    if dim_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        rep = dim_check(claims_data, pdf_path, max_vision_calls=2, client=client, field=field_top)
        save_dim(rep)
        if getattr(rep, "gate_reason", ""):
            console.print(f"       [dim]{rep.gate_reason}[/dim]")
        else:
            console.print(f"       {rep.n_findings} findings")

    console.print("[cyan]4/10[/cyan] Symbolic verification")
    sym_path = get_path("processed_data") / "sympy_reports" / f"{doi}.json"
    if sym_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        rep = verify_paper(claims_data, pdf_path, max_claims=8, client=client)
        save_sym(rep)
        console.print(f"       verified={rep.n_verified}, refuted={rep.n_refuted}, "
                      f"unable={rep.n_unable}")

    console.print("[cyan]5/10[/cyan] Novelty / literature")
    nov_path = get_path("processed_data") / "novelty_reports" / f"{doi}.json"
    if nov_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        rep = analyze_novelty(doi, client=client)
        save_nov(rep)
        console.print(f"       {rep.n_related_retrieved} related, {rep.n_findings} findings")

    console.print("[cyan]6/10[/cyan] Structure & narrative")
    str_path = get_path("processed_data") / "structure_reports" / f"{doi}.json"
    if str_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        rep = analyze_structure(doi, client=client)
        save_str(rep)
        console.print(f"       {rep.n_findings} findings")

    console.print("[cyan]7/10[/cyan] Domain expert")
    dom_path = get_path("processed_data") / "domain_reports" / f"{doi}.json"
    if dom_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        rep = analyze_domain(doi, client=client)
        save_dom(rep)
        console.print(f"       persona={rep.persona} findings={rep.n_findings}")

    console.print("[cyan]8/10[/cyan] Reproducibility")
    repro_path = get_path("processed_data") / "reproducibility_reports" / f"{doi}.json"
    if repro_path.exists():
        console.print("       [dim]cached[/dim]")
    else:
        rep = analyze_reproducibility(doi, client=client)
        save_repro(rep)
        console.print(f"       scrutiny={rep.scrutiny_level} findings={rep.n_findings}")

    console.print("[cyan]9/10[/cyan] Math aggregation")
    aggregate_and_render(doi)

    console.print("[cyan]10/10[/cyan] Full referee section + triage score")
    json_path, md_path = aggregate_and_render_full(doi)

    triage_path = get_path("processed_data") / "triage_scores" / f"{doi}.json"
    triage = json.loads(triage_path.read_text()) if triage_path.exists() else {}
    verdict = triage.get("verdict", "unknown")
    score = triage.get("score", 0)
    color = {"priority": "green", "standard": "yellow", "reject": "red"}.get(verdict, "white")

    # Optional LLM-authorship policy check (opt-in; does not affect score).
    llm_verdict = None
    llm_confidence = None
    if check_llm:
        console.print("[cyan]Policy[/cyan]  LLM-authorship detection")
        llm_path = get_path("processed_data") / "llm_detection_reports" / f"{doi}.json"
        if llm_path.exists():
            console.print("        [dim]cached[/dim]")
        else:
            rep = analyze_llm_authorship(doi)
            save_llm(rep)
            console.print(f"        verdict={rep.verdict}  confidence={rep.confidence:.2f}  "
                          f"signals={rep.n_signals} (strong={rep.n_strong}, mod={rep.n_moderate})")
        if llm_path.exists():
            llm_data = json.loads(llm_path.read_text())
            llm_verdict = llm_data.get("verdict")
            llm_confidence = llm_data.get("confidence")
            # Re-render markdown so the Policy section is embedded in the report.
            aggregate_and_render_full(doi)

    console.rule("[bold green]Done[/bold green]")
    console.print(f"Triage:   [{color}]{verdict.upper()}[/{color}]  (score {score}/100)")
    if llm_verdict:
        llm_color = {"likely_human": "green", "unclear": "white",
                     "possibly_llm": "yellow", "likely_llm": "red"}.get(llm_verdict, "white")
        cconf = f"{llm_confidence:.2f}" if llm_confidence is not None else "?"
        console.print(f"Policy:   [{llm_color}]{llm_verdict.upper()}[/{llm_color}]  (confidence {cconf})")
    console.print(f"Elapsed:  {time.time()-total_t0:.1f}s")
    console.print(f"Report:   {md_path}")

    _refresh_dashboard()
    dashboard = get_path("processed_data") / "editorial_dashboard.html"
    console.print(f"Dashboard: {dashboard}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, help="Local PDF path")
    ap.add_argument("--arxiv", type=str,
                    help="arXiv id or URL (e.g. 2401.12345)")
    ap.add_argument("--url", type=str,
                    help="Any HTTP(S) URL that returns a PDF (ChemRxiv, bioRxiv, medRxiv, publisher pages, personal sites).")
    ap.add_argument("--doi", type=str,
                    help="Explicit identifier (defaults to filename stem when using --pdf)")
    ap.add_argument("--check-llm", action="store_true",
                    help="Also run LLM-authorship detection (opt-in policy check; "
                         "does not affect the triage score).")
    args = ap.parse_args()

    doi, pdf_path = resolve_input(args)
    run_pipeline(doi, pdf_path, check_llm=args.check_llm)


if __name__ == "__main__":
    main()
