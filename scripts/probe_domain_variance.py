"""Domain agent variance probe with content-preserving storage.

Reruns domain agent 3 times per paper and saves each run's full findings
to variance_probe/domain/{doi}_run{n}.json. Then analyzes with three
signature strategies to answer the key question:

  Is the model finding the SAME 3 concerns with different wording (fixable),
  or DIFFERENT 3 concerns each run (fabrication / quota-filling)?

Signature strategies:
  loose  : (location,)                            same LOCATION each run?
  medium : (location, kind)                       same CONCEPT same LOCATION?
  strict : (location, kind, evidence[:80])        same LITERAL text?

Interpretation:
  loose high, strict low  → wording drift, fixable via signature or prompt
  loose low               → model picks different concerns each run
                            (this is the fabrication hypothesis; harder fix)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.gemini_client import GeminiClient
from peerlens.agents.domain_expert import analyze_domain, save_report as save_dom
from peerlens.settings import get_path
from rich.console import Console

console = Console()

DOIS = [
    "SciPostPhys.20.3.080",
    "SciPostPhys.20.4.097",
    "SciPostPhys.21.1.007",
    "SciPostPhys.20.6.166",
    "SciPostPhys.20.4.119",
    "SciPostPhys.21.1.011",
]

N_RUNS = 3
PROBE_DIR = get_path("processed_data") / "variance_probe" / "domain"


def all_probes_exist() -> bool:
    for doi in DOIS:
        for i in range(N_RUNS):
            if not (PROBE_DIR / f"{doi}_run{i+1}.json").exists():
                return False
    return True


def rerun_and_save(doi: str, run_idx: int, client: GeminiClient) -> int:
    cache_path = get_path("processed_data") / "domain_reports" / f"{doi}.json"
    cache_path.unlink(missing_ok=True)
    rep = analyze_domain(doi, client=client)
    save_dom(rep)
    data = json.loads(cache_path.read_text())

    probe_path = PROBE_DIR / f"{doi}_run{run_idx+1}.json"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return len(data.get("findings", []))


def norm_loc(loc: str) -> str:
    """Normalize location for cross-run comparison.

    'Abstract', 'Abstract, sentence 2', 'Abstract sentence 2' → 'abstract'
    'Section 3.1', 'Sec 3.1', 'Sec. 3.1' → 'section 3.1'
    """
    import re
    if not loc:
        return "(unspecified)"
    s = loc.strip().lower()
    # Strip "sentence N", "line N", "paragraph N" refinements
    s = re.sub(r"[,;]?\s*(sentence|line|paragraph|para\.?)\s+\d+", "", s)
    # Normalize section refs
    s = re.sub(r"\bsec\.?\b", "section", s)
    # Trim internal double spaces + trailing punctuation
    s = re.sub(r"\s+", " ", s).strip(" .,;:")
    return s


def analyze_paper(doi: str) -> dict:
    findings_per_run = []
    for i in range(N_RUNS):
        probe_path = PROBE_DIR / f"{doi}_run{i+1}.json"
        data = json.loads(probe_path.read_text())
        findings_per_run.append(data.get("findings", []))

    loose_sigs = [
        {norm_loc(f.get("location", "")) for f in fs} for fs in findings_per_run
    ]
    medium_sigs = [
        {(norm_loc(f.get("location", "")), f.get("kind", "")) for f in fs}
        for fs in findings_per_run
    ]
    strict_sigs = [
        {(norm_loc(f.get("location", "")),
          f.get("kind", ""),
          (f.get("evidence", "") or "")[:80].strip().lower()) for f in fs}
        for fs in findings_per_run
    ]

    def stability(sigs_list):
        all_ = set().union(*sigs_list) if sigs_list else set()
        stable = set.intersection(*sigs_list) if sigs_list else set()
        pct = len(stable) / len(all_) * 100 if all_ else 100.0
        return len(stable), len(all_), pct

    n_unspec = sum(
        1 for fs in findings_per_run for f in fs
        if not (f.get("location") or "").strip()
    )

    return {
        "counts": [len(fs) for fs in findings_per_run],
        "n_unspec_location": n_unspec,
        "loose": stability(loose_sigs),
        "medium": stability(medium_sigs),
        "strict": stability(strict_sigs),
        "findings_per_run": findings_per_run,
    }


def print_paper(doi: str, res: dict) -> None:
    ls, lt, lp = res["loose"]
    ms, mt, mp = res["medium"]
    ss, st, sp = res["strict"]
    console.print(f"[bold]{doi}[/bold]  counts: {res['counts']}  "
                  f"unspec_loc={res['n_unspec_location']}")
    console.print(f"  location-only      {ls}/{lt} = {lp:.0f}%")
    console.print(f"  location+kind      {ms}/{mt} = {mp:.0f}%")
    console.print(f"  loc+kind+evidence  {ss}/{st} = {sp:.0f}%")

    for i, fs in enumerate(res["findings_per_run"]):
        console.print(f"  [dim]run {i+1}:[/dim]")
        for f in fs:
            loc = (f.get("location") or "(unspec)")[:40]
            kind = (f.get("kind") or "")[:20]
            sev = (f.get("severity") or "")[:8]
            ev = (f.get("evidence") or "")[:70]
            console.print(f"    [{sev:<8}] {kind:<20} @ {loc:<40} :: {ev}")
    console.print()


def main():
    if not all_probes_exist():
        console.rule("[bold cyan]Collecting 6 papers × 3 runs — domain only[/bold cyan]")
        client = GeminiClient()
        t0 = time.time()
        for doi in DOIS:
            console.print(f"[cyan]{doi}[/cyan]")
            for i in range(N_RUNS):
                n = rerun_and_save(doi, i, client)
                console.print(f"  run {i+1}: {n} findings saved")
        console.print(f"[dim]elapsed: {time.time()-t0:.0f}s[/dim]")
    else:
        console.rule("[bold cyan]Cached probe found — skipping re-run[/bold cyan]")

    console.rule("[bold cyan]Analysis[/bold cyan]")
    console.print()
    per_paper = {}
    for doi in DOIS:
        res = analyze_paper(doi)
        per_paper[doi] = res
        print_paper(doi, res)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    loose_pcts = [per_paper[d]["loose"][2] for d in DOIS]
    medium_pcts = [per_paper[d]["medium"][2] for d in DOIS]
    strict_pcts = [per_paper[d]["strict"][2] for d in DOIS]

    console.rule("[bold green]Summary[/bold green]")
    console.print(f"  location-only mean stability     : {mean(loose_pcts):.0f}%")
    console.print(f"  location+kind mean stability     : {mean(medium_pcts):.0f}%")
    console.print(f"  loc+kind+evidence mean stability : {mean(strict_pcts):.0f}%")
    console.print()
    console.print("[bold]Diagnosis[/bold]")
    if mean(loose_pcts) > 55 and mean(strict_pcts) < 25:
        console.print("  → wording drift dominant. Fix: normalized signature "
                      "in scoring/aggregation, or tighten prompt for consistent phrasing.")
    elif mean(loose_pcts) < 35:
        console.print("  → fabrication dominant. Fix: prompt overhaul + schema "
                      "restriction (silence-is-default), reproducibility-style gate.")
    else:
        console.print("  → mixed. Both effects present; fix must address both.")


if __name__ == "__main__":
    main()
