"""Run referee comparison across multiple papers, print a dashboard."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anthropic import Anthropic
from peerlens.evaluation.referee_comparison import compare, save_comparison
from peerlens.settings import get_path
from rich.console import Console
from rich.table import Table

console = Console()

DOIS = [
    "SciPostPhys.20.3.080",  # grav/cosmo
    "SciPostPhys.20.4.097",  # grav/cosmo
    "SciPostPhys.21.1.007",  # math phys
    "SciPostPhys.20.6.166",  # stat/soft
    "SciPostPhys.20.4.119",  # hep theory — math empty but novelty works
    "SciPostPhys.21.1.011",  # hero paper
]

client = Anthropic()
results = []
total_cost = 0.0

for doi in DOIS:
    console.rule(f"[bold]{doi}[/bold]")
    try:
        res = compare(doi, client=client)
        save_comparison(res)
        results.append(res)
        cost = (res.input_tokens/1e6)*3.00 + (res.output_tokens/1e6)*15.00
        total_cost += cost
        console.print(f"  ai={res.n_ai_findings}  ref={res.n_referee_math_concerns}  "
                      f"match={res.n_matches}  missed={res.n_missed}  novel={res.n_novel}  "
                      f"[dim]${cost:.3f}[/dim]")
    except Exception as e:
        console.print(f"  [red]FAILED[/red]: {type(e).__name__}: {str(e)[:150]}")
    time.sleep(1)

console.rule("[bold cyan]Dashboard[/bold cyan]")
tbl = Table(show_lines=True)
tbl.add_column("DOI", style="cyan", max_width=22)
tbl.add_column("Verdict", style="yellow", max_width=8)
tbl.add_column("AI find", justify="right")
tbl.add_column("Ref concerns", justify="right")
tbl.add_column("Match", justify="right", style="green")
tbl.add_column("Missed", justify="right", style="red")
tbl.add_column("Novel", justify="right", style="yellow")
tbl.add_column("Precision", justify="right")
tbl.add_column("Recall", justify="right")

import json
totals = {"ai": 0, "ref": 0, "match": 0, "missed": 0, "novel": 0}
for res in results:
    ai = res.n_ai_findings
    ref = res.n_referee_math_concerns
    match = res.n_matches
    missed = res.n_missed
    novel = res.n_novel

    prec = match / ai if ai else 0.0
    rec = match / ref if ref else 0.0

    # get verdict
    verdict_path = get_path("processed_data") / "math_reports" / f"{res.doi_label}.json"
    verdict = json.loads(verdict_path.read_text()).get("verdict", "?")

    tbl.add_row(
        res.doi_label,
        verdict,
        str(ai), str(ref), str(match), str(missed), str(novel),
        f"{prec:.0%}", f"{rec:.0%}",
    )
    for k in totals: totals[k] = 0
    totals["ai"] += ai; totals["ref"] += ref
    totals["match"] += match; totals["missed"] += missed; totals["novel"] += novel

# aggregate row
agg_ai = sum(r.n_ai_findings for r in results)
agg_ref = sum(r.n_referee_math_concerns for r in results)
agg_match = sum(r.n_matches for r in results)
agg_missed = sum(r.n_missed for r in results)
agg_novel = sum(r.n_novel for r in results)
agg_prec = agg_match / agg_ai if agg_ai else 0
agg_rec = agg_match / agg_ref if agg_ref else 0

tbl.add_row(
    "[bold]TOTAL[/bold]", "-",
    f"[bold]{agg_ai}[/bold]", f"[bold]{agg_ref}[/bold]",
    f"[bold]{agg_match}[/bold]", f"[bold]{agg_missed}[/bold]", f"[bold]{agg_novel}[/bold]",
    f"[bold]{agg_prec:.0%}[/bold]", f"[bold]{agg_rec:.0%}[/bold]",
)

console.print(tbl)
console.print(f"\n[dim]Total cost: ${total_cost:.3f}[/dim]")

# Per-paper novel_findings assessment breakdown
console.rule("[bold]Novel findings — false positive risk by paper[/bold]")
for res in results:
    console.print(f"\n[bold]{res.doi_label}[/bold]")
    from collections import Counter
    assess = Counter(n.get("assessment", "?") for n in res.novel)
    for k, v in assess.items():
        console.print(f"  {k}: {v}")
