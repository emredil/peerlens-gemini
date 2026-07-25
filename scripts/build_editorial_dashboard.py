"""Generate an HTML editorial dashboard from all triage scores on disk.

Renders one HTML page with three sections (priority / standard / reject),
each a sortable table, and a link to the per-paper markdown report.
"""
import json
import sys
from pathlib import Path
from html import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.settings import get_path


def load_all_triages() -> list[dict]:
    scores_dir = get_path("processed_data") / "triage_scores"
    out = []
    for p in sorted(scores_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            out.append(d)
        except Exception:
            continue
    return out


def load_parsed_title(doi: str) -> str:
    parsed = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not parsed.exists():
        return ""
    try:
        return json.loads(parsed.read_text()).get("title", "")
    except Exception:
        return ""


def load_classification(doi: str) -> dict:
    path = get_path("processed_data") / "classifications" / f"{doi}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


BUCKETS = [
    ("priority", "Priority (score ≥ 80)", "#0a7c00"),
    ("standard", "Standard (40–79)", "#a06000"),
    ("reject",   "Reject candidates (score < 40)", "#a00000"),
]


HTML_HEAD = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>PeerLens — Editorial triage dashboard</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 40px auto; max-width: 1150px;
         color: #222; background: #fafafa; }
  h1 { margin-bottom: 0; }
  .meta { color: #666; margin-top: 4px; margin-bottom: 30px; }
  h2 { padding: 8px 12px; color: white; border-radius: 4px; margin-top: 40px; }
  table { border-collapse: collapse; width: 100%; margin-top: 10px; background: white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  th, td { padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }
  th { background: #f2f2f2; }
  td.score { font-family: 'SF Mono', 'Menlo', monospace; font-weight: 600; text-align: right; }
  td.doi { font-family: 'SF Mono', 'Menlo', monospace; font-size: 0.9em; }
  ul.topneg { margin: 0; padding-left: 18px; font-size: 0.85em; color: #444; }
  ul.topneg li { margin-bottom: 3px; }
  .weight { font-family: monospace; color: #a00000; font-weight: 600; }
  .empty { color: #999; padding: 10px; }
  .subdomain { font-size: 0.8em; color: #666; }
  a.md-link { text-decoration: none; color: #005ea6; }
  a.md-link:hover { text-decoration: underline; }
</style>
</head>
<body>
"""

HTML_TAIL = """
</body>
</html>
"""


def render_row(t: dict) -> str:
    doi = t.get("doi_label", "?")
    score = t.get("score", 0)
    title = load_parsed_title(doi)
    cls = load_classification(doi)
    subs = ", ".join(cls.get("subdomains", []) or [])

    top_neg_html = ""
    top = t.get("top_negative", [])
    if top:
        items = []
        for c in top:
            w = c.get("weight", 0)
            src = c.get("source", "")
            kind = c.get("kind", "")
            sev = c.get("severity", "")
            loc = c.get("location", "") or "—"
            ev = c.get("evidence_short", "")
            items.append(
                f"<li><span class='weight'>{w:+d}</span> "
                f"<b>{escape(src)}.{escape(kind)}</b> [{escape(sev)}] @ {escape(loc)}"
                f"<br><span style='color:#555'>{escape(ev)}</span></li>"
            )
        top_neg_html = "<ul class='topneg'>" + "".join(items) + "</ul>"

    md_rel = f"referee_sections/{doi}_full.md"

    return (
        "<tr>"
        f"<td class='score'>{score}</td>"
        f"<td class='doi'>{escape(doi)}<br><span class='subdomain'>{escape(subs)}</span></td>"
        f"<td>{escape(title)}<br>{top_neg_html}</td>"
        f"<td><a class='md-link' href='{md_rel}'>report.md</a></td>"
        "</tr>"
    )


def build_dashboard(out_path: Path) -> Path:
    triages = load_all_triages()
    triages.sort(key=lambda t: -t.get("score", 0))

    n_total = len(triages)
    n_priority = sum(1 for t in triages if t.get("verdict") == "priority")
    n_standard = sum(1 for t in triages if t.get("verdict") == "standard")
    n_reject = sum(1 for t in triages if t.get("verdict") == "reject")
    n_insufficient = n_total - n_priority - n_standard - n_reject

    parts = [HTML_HEAD]
    parts.append(f"<h1>PeerLens — Editorial triage</h1>")
    parts.append(
        f"<p class='meta'>{n_total} papers processed. "
        f"Priority: {n_priority} · Standard: {n_standard} · Reject: {n_reject}"
        + (f" · Insufficient: {n_insufficient}" if n_insufficient else "") + "</p>"
    )

    for verdict_key, verdict_label, color in BUCKETS:
        bucket = [t for t in triages if t.get("verdict") == verdict_key]
        parts.append(f"<h2 style='background:{color}'>{escape(verdict_label)} — {len(bucket)}</h2>")
        if not bucket:
            parts.append("<div class='empty'>Empty.</div>")
            continue
        parts.append("<table>")
        parts.append("<tr><th style='width:70px'>Score</th>"
                     "<th style='width:220px'>DOI / subdomain</th>"
                     "<th>Title &amp; top signals</th>"
                     "<th style='width:90px'>Report</th></tr>")
        for t in bucket:
            parts.append(render_row(t))
        parts.append("</table>")

    if n_insufficient:
        bucket = [t for t in triages
                  if t.get("verdict") not in ("priority", "standard", "reject")]
        parts.append(f"<h2 style='background:#555'>Insufficient data — {len(bucket)}</h2>")
        parts.append("<table>")
        parts.append("<tr><th>Score</th><th>DOI</th><th>Title</th><th>Report</th></tr>")
        for t in bucket:
            parts.append(render_row(t))
        parts.append("</table>")

    parts.append(HTML_TAIL)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


def main():
    out = get_path("processed_data") / "editorial_dashboard.html"
    path = build_dashboard(out)
    print(f"Dashboard written: {path}")


if __name__ == "__main__":
    main()
