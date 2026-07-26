"""End-to-end score variance probe.

Reruns the full pipeline 3 times for each paper (clearing all Gemini-driven
caches between runs) and records the final triage score. Answers: 'is the
score deterministic in practice after domain+structure fixes?'
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.gemini_client import GeminiClient
from peerlens.agents.claim_extractor import extract_claims, save_claims
from peerlens.agents.novelty_agent import analyze_novelty, save_report as save_nov
from peerlens.agents.structure_agent import analyze_structure, save_report as save_str
from peerlens.agents.domain_expert import analyze_domain, save_report as save_dom
from peerlens.agents.reproducibility_agent import analyze_reproducibility, save_report as save_repro
from peerlens.orchestrator.math_aggregator import aggregate_and_render
from peerlens.orchestrator.referee_aggregator import aggregate_and_render_full
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

# Reset these each run; keep parsed/claims/dimensional/sympy cached (deterministic).
RESET_FOLDERS = [
    "novelty_reports", "structure_reports", "domain_reports",
    "reproducibility_reports", "referee_reports", "referee_sections",
    "math_reports", "triage_scores",
]


def clear_caches(doi):
    for folder in RESET_FOLDERS:
        d = get_path("processed_data") / folder
        for ext in (".json", ".md", "_full.md"):
            p = d / f"{doi}{ext}"
            p.unlink(missing_ok=True)


def one_pipeline_run(doi, client):
    # Assume claims/dimensional/sympy already cached from previous full runs.
    # If not, they'll be regenerated (deterministic).

    parsed = json.loads((get_path("processed_data") / "parsed" / f"{doi}.json").read_text())
    claims_path = get_path("processed_data") / "claims" / f"{doi}.json"
    if not claims_path.exists():
        res = extract_claims(parsed, client=client)
        save_claims(res)

    # 4 Gemini agents (variable output)
    save_nov(analyze_novelty(doi, client=client))
    save_str(analyze_structure(doi, client=client))
    save_dom(analyze_domain(doi, client=client))
    save_repro(analyze_reproducibility(doi, client=client))

    aggregate_and_render(doi)
    aggregate_and_render_full(doi)

    triage = json.loads((get_path("processed_data") / "triage_scores" / f"{doi}.json").read_text())
    return triage.get("score", 0), triage.get("verdict", "?")


def main():
    console.rule("[bold cyan]Full-pipeline score variance — 6 × 3[/bold cyan]")
    client = GeminiClient()

    all_results = {}
    t0 = time.time()
    for doi in DOIS:
        console.print(f"[cyan]{doi}[/cyan]")
        scores = []
        for i in range(N_RUNS):
            clear_caches(doi)
            score, verdict = one_pipeline_run(doi, client)
            scores.append((score, verdict))
            console.print(f"  run {i+1}: {score:3d}  {verdict}")
        smin = min(s[0] for s in scores)
        smax = max(s[0] for s in scores)
        spread = smax - smin
        verdicts = {s[1] for s in scores}
        stable_verdict = "YES" if len(verdicts) == 1 else "NO"
        console.print(f"  [dim]spread={spread}, verdict_stable={stable_verdict}[/dim]")
        all_results[doi] = {"scores": scores, "spread": spread, "verdict_stable": stable_verdict}

    console.print(f"[dim]elapsed: {time.time()-t0:.0f}s[/dim]")

    console.rule("[bold green]Summary[/bold green]")
    max_spread = max(r["spread"] for r in all_results.values())
    unstable_verdicts = [d for d, r in all_results.items() if r["verdict_stable"] == "NO"]
    console.print(f"Max score spread across 3 runs: [bold]{max_spread}[/bold] points")
    if unstable_verdicts:
        console.print(f"[red]Papers with unstable verdicts:[/red] {', '.join(unstable_verdicts)}")
    else:
        console.print("[green]All verdicts stable across 3 runs.[/green]")

    if max_spread <= 5 and not unstable_verdicts:
        console.print("\n[bold green]DIAGNOSIS[/bold green]: Score is effectively deterministic.")
        console.print("Hash-based caching is optional, not required.")
    elif max_spread <= 15:
        console.print("\n[bold yellow]DIAGNOSIS[/bold yellow]: Moderate variance; hash cache "
                      "recommended for editor confidence.")
    else:
        console.print("\n[bold red]DIAGNOSIS[/bold red]: Large variance; hash cache required.")

    out = get_path("processed_data") / "score_variance_probe.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    console.print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
