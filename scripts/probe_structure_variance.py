"""Structure agent variance probe with content-preserving storage."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.gemini_client import GeminiClient
from peerlens.agents.structure_agent import analyze_structure, save_report as save_str
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
PROBE_DIR = get_path("processed_data") / "variance_probe" / "structure"


def all_probes_exist() -> bool:
    for doi in DOIS:
        for i in range(N_RUNS):
            if not (PROBE_DIR / f"{doi}_run{i+1}.json").exists():
                return False
    return True


def rerun_and_save(doi, run_idx, client):
    cache = get_path("processed_data") / "structure_reports" / f"{doi}.json"
    cache.unlink(missing_ok=True)
    rep = analyze_structure(doi, client=client)
    save_str(rep)
    data = json.loads(cache.read_text())
    probe = PROBE_DIR / f"{doi}_run{run_idx+1}.json"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def norm_trigger(t):
    return (t or "").strip().lower()


def analyze(doi):
    fpr = []
    for i in range(N_RUNS):
        p = PROBE_DIR / f"{doi}_run{i+1}.json"
        fpr.append(json.loads(p.read_text()).get("findings", []))

    # Signature: (trigger_phrase, kind) — trigger is deterministic from stage 1
    sigs = [
        {(norm_trigger(f.get("trigger_phrase", "")), f.get("kind", "")) for f in fs}
        for fs in fpr
    ]

    def stability(sl):
        u = set().union(*sl) if sl else set()
        s = set.intersection(*sl) if sl else set()
        return len(s), len(u), (len(s) / len(u) * 100 if u else 100.0)

    return {
        "counts": [len(fs) for fs in fpr],
        "candidates": json.loads((PROBE_DIR / f"{doi}_run1.json").read_text()).get("n_candidates_scanned", 0),
        "stab": stability(sigs),
        "findings_per_run": fpr,
    }


def main():
    if not all_probes_exist():
        console.rule("[bold cyan]Structure probe — 6 papers × 3 runs[/bold cyan]")
        client = GeminiClient()
        t0 = time.time()
        for doi in DOIS:
            console.print(f"[cyan]{doi}[/cyan]")
            for i in range(N_RUNS):
                d = rerun_and_save(doi, i, client)
                console.print(f"  run {i+1}: cand={d.get('n_candidates_scanned',0)}  "
                              f"found={len(d.get('findings',[]))}")
        console.print(f"[dim]elapsed: {time.time()-t0:.0f}s[/dim]")
    else:
        console.rule("[bold cyan]Cached probe — skipping re-run[/bold cyan]")

    console.rule("[bold cyan]Analysis[/bold cyan]")
    per_paper = {}
    for doi in DOIS:
        r = analyze(doi)
        per_paper[doi] = r
        s, u, p = r["stab"]
        console.print(f"[bold]{doi}[/bold]  candidates={r['candidates']:<3}  "
                      f"counts={r['counts']}  stability={s}/{u}={p:.0f}%")
        for i, fs in enumerate(r["findings_per_run"]):
            for f in fs:
                trig = (f.get("trigger_phrase") or "")[:30]
                kind = (f.get("kind") or "")[:20]
                sev = (f.get("severity") or "")[:8]
                ev = (f.get("evidence") or "")[:70]
                console.print(f"  [dim]run {i+1}[/dim]  [{sev:<8}] {kind:<20} "
                              f"trigger=\"{trig}\" :: {ev}")
        console.print()

    pcts = [per_paper[d]["stab"][2] for d in DOIS]
    mean = sum(pcts) / len(pcts)
    console.rule("[bold green]Summary[/bold green]")
    console.print(f"Mean stability (trigger+kind): {mean:.0f}%")


if __name__ == "__main__":
    main()
