"""Render a dimensional check into a SciPost-style referee section.

This produces markdown that will slot into the full synthesized referee
report at the end of Phase 1a.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.settings import get_path
from rich.console import Console

console = Console()

doi = "SciPostPhys.21.1.011"
report_path = get_path("processed_data") / "dimensional_reports" / f"{doi}.json"
report = json.loads(report_path.read_text())

lines = []
lines.append(f"# Dimensional Analysis — {report['doi_label']}")
lines.append("")
lines.append(f"**Summary:** {report['n_findings']} findings — "
             f"{report['n_critical']} critical, {report['n_warning']} warning, {report['n_info']} info.")
lines.append("")

# Group by severity
by_sev = {"critical": [], "warning": [], "info": []}
for f in report["findings"]:
    by_sev.setdefault(f["severity"], []).append(f)

if by_sev["critical"]:
    lines.append("## Critical dimensional issues")
    lines.append("")
    for f in by_sev["critical"]:
        lines.append(f"- **{f['location']}** — {f['verdict']}. {f['evidence']}")
        if f.get("suggested_fix"):
            lines.append(f"  *Suggested fix:* {f['suggested_fix']}")
        lines.append("")

if by_sev["warning"]:
    lines.append("## Points requiring author clarification")
    lines.append("")
    for f in by_sev["warning"]:
        lines.append(f"- **{f['location']}** — {f['verdict']}. {f['evidence']}")
        if f.get("suggested_fix"):
            lines.append(f"  *Suggested fix:* {f['suggested_fix']}")
        lines.append("")

if by_sev["info"] and not (by_sev["critical"] or by_sev["warning"]):
    lines.append("## Notes")
    lines.append("")
    lines.append("No dimensional errors were detected. The paper uses consistent conventions throughout:")
    lines.append("")
    for f in by_sev["info"][:5]:  # cap to top 5
        lines.append(f"- {f['location']}: {f['verdict']}. "
                     f"{f['evidence'][:200]}{'...' if len(f['evidence']) > 200 else ''}")
elif by_sev["info"]:
    lines.append("## Passed checks (summary)")
    lines.append("")
    lines.append(f"{len(by_sev['info'])} dimensional claims were checked and found consistent, "
                 "listed abbreviated below.")
    lines.append("")
    for f in by_sev["info"][:5]:
        lines.append(f"- {f['location']}: {f['verdict']}")

md = "\n".join(lines)

out_dir = get_path("processed_data") / "referee_sections"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{doi}_dimensional.md"
out_path.write_text(md, encoding="utf-8")

console.print(f"[green]Rendered:[/green] {out_path}")
console.print("\n" + md)
