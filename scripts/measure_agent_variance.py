"""Measure non-determinism and domain -8 pattern across 3 reruns.

Reruns each Gemini-driven agent 3 times per paper, forcing cache miss
by clearing the specific report file before each run. Records per-agent:
  - number of findings each run
  - variance across runs
  - which findings are stable vs which are new

Only clears the reports for the agents we're measuring; upstream stages
(claims, dimensional, sympy) stay cached to save cost and time.
"""
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.gemini_client import GeminiClient
from peerlens.agents.novelty_agent import analyze_novelty, save_report as save_nov
from peerlens.agents.structure_agent import analyze_structure, save_report as save_str
from peerlens.agents.domain_expert import analyze_domain, save_report as save_dom
from peerlens.agents.reproducibility_agent import analyze_reproducibility, save_report as save_repro
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

AGENTS = [
    ("novelty",         "novelty_reports",         analyze_novelty,        save_nov),
    ("structure",       "structure_reports",       analyze_structure,      save_str),
    ("domain",          "domain_reports",          analyze_domain,         save_dom),
    ("reproducibility", "reproducibility_reports", analyze_reproducibility, save_repro),
]

N_RUNS = 3


def rerun_agent(agent_name, folder, fn, save_fn, doi, client):
    """Clear cache, rerun agent, return findings list + severity counts."""
    path = get_path("processed_data") / folder / f"{doi}.json"
    path.unlink(missing_ok=True)
    rep = fn(doi, client=client)
    save_fn(rep)
    data = json.loads(path.read_text())
    findings = data.get("findings", [])
    counts = {"critical": 0, "warning": 0, "info": 0}
    for f in findings:
        sev = f.get("severity", "info")
        counts[sev] = counts.get(sev, 0) + 1
    return findings, counts


def signature(f):
    """A fingerprint we can compare across runs. Location + kind + first 60
    chars of evidence — captures "same concern" even if wording drifts."""
    return (
        (f.get("location", "") or "").strip()[:60],
        (f.get("kind", "") or "").strip(),
        (f.get("evidence", "") or "").strip()[:60].lower(),
    )


def main():
    client = GeminiClient()
    results = defaultdict(lambda: defaultdict(list))

    t0 = time.time()
    for doi in DOIS:
        console.rule(f"[bold]{doi}[/bold]")
        for agent_name, folder, fn, save_fn in AGENTS:
            counts_per_run = []
            signatures_per_run = []
            for run in range(N_RUNS):
                findings, counts = rerun_agent(agent_name, folder, fn, save_fn, doi, client)
                counts_per_run.append(counts)
                signatures_per_run.append({signature(f) for f in findings})
                total = sum(counts.values())
                console.print(f"  {agent_name:<18} run {run+1}: "
                              f"n={total:<3}  "
                              f"crit={counts['critical']} warn={counts['warning']} info={counts['info']}")

            all_sigs = set().union(*signatures_per_run)
            stable = set.intersection(*signatures_per_run)
            n_stable = len(stable)
            n_unique = len(all_sigs)
            stability = (n_stable / n_unique * 100) if n_unique else 100.0
            console.print(f"    [dim]{agent_name}: {n_stable}/{n_unique} stable "
                          f"({stability:.0f}%), "
                          f"totals over 3 runs: {[sum(c.values()) for c in counts_per_run]}[/dim]")

            results[doi][agent_name] = {
                "counts_per_run": counts_per_run,
                "n_stable": n_stable,
                "n_unique": n_unique,
                "stability_pct": stability,
                "totals": [sum(c.values()) for c in counts_per_run],
            }

    out = get_path("processed_data") / "variance_measurement.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    console.rule("[bold green]Done[/bold green]")
    console.print(f"Elapsed: {time.time()-t0:.0f}s")
    console.print(f"Written: {out}")

    # Summary — which agents are worst offenders?
    console.print()
    console.print("[bold]Per-agent stability summary (mean across all papers)[/bold]")
    agent_totals = defaultdict(list)
    for doi in results:
        for agent in results[doi]:
            agent_totals[agent].append(results[doi][agent]["stability_pct"])
    for agent, pcts in agent_totals.items():
        mean = sum(pcts) / len(pcts)
        console.print(f"  {agent:<18} mean stability: {mean:.0f}%   "
                      f"(per-paper: {[f'{p:.0f}%' for p in pcts]})")


if __name__ == "__main__":
    main()
