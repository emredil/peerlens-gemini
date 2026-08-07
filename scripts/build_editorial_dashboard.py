"""Generate an HTML editorial dashboard from all domain-expert verdicts on disk.

Renders one HTML page with four sections (accept / minor revision /
major revision / reject), each a sortable table, and a link to the
per-paper markdown report.
"""
import json
import sys
from pathlib import Path
from html import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.settings import get_path


VERDICT_ORDER = ["accept", "minor revision", "major revision", "reject", "unknown"]

VERDICT_META = {
    "accept":          ("Accept",          "#0a7c00"),
    "minor revision":  ("Minor revision",  "#a06000"),
    "major revision":  ("Major revision",  "#c05400"),
    "reject":          ("Reject",          "#a00000"),
    "unknown":         ("Unclassified",    "#555555"),
}


def load_all_verdicts() -> list[dict]:
    """Read domain_reports/*.json for every paper on disk, one row each."""
    dom_dir = get_path("processed_data") / "domain_reports"
    out = []
    for p in sorted(dom_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        doi = d.get("doi_label") or p.stem
        verdict = (d.get("verdict") or "").strip().lower() or "unknown"
        if verdict not in VERDICT_META:
            verdict = "unknown"
        rationale = (d.get("verdict_rationale") or "").strip()
        out.append({
            "doi_label": doi,
            "verdict": verdict,
            "rationale": rationale,
            "subdomain": d.get("subdomain_used", "") or "",
            "persona": d.get("persona", "") or "",
        })
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


def load_policy(doi: str) -> dict:
    path = get_path("processed_data") / "llm_detection_reports" / f"{doi}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _policy_badge(policy: dict) -> str:
    if not policy:
        return ""
    verdict = policy.get("verdict", "unclear")
    try:
        conf = float(policy.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    palette = {
        "likely_human": ("#e6f4ea", "#0a7c00", "✓ HUMAN"),
        "unclear":      ("#f0f0f0", "#555555", "? UNCLEAR"),
        "possibly_llm": ("#fff3d9", "#8a5a00", "⚠ POSSIBLE AI"),
        "likely_llm":   ("#fde2e2", "#a00000", "⚠ LIKELY AI"),
    }
    bg, fg, label = palette.get(verdict, ("#f0f0f0", "#555555", str(verdict).upper()))
    return (
        f'<span style="display:inline-block; padding:2px 8px; margin-left:6px; '
        f'background:{bg}; color:{fg}; border-radius:10px; font-size:0.75em; '
        f'font-weight:600;">{escape(label)} · {conf:.0%}</span>'
    )


def _row(entry: dict) -> str:
    doi = entry["doi_label"]
    title = load_parsed_title(doi) or "(untitled)"
    cls = load_classification(doi)
    field = cls.get("field_top", "")
    subdomain = entry["subdomain"] or (cls.get("subdomains", [""])[0] if cls.get("subdomains") else "")
    persona = entry["persona"]
    rationale = entry["rationale"]

    policy_badge = _policy_badge(load_policy(doi))

    report_link = f'referee_sections/{doi}_full.md'

    field_tag = f'<span class="tag">{escape(field)}</span>' if field else ""
    sub_tag = f'<span class="tag muted">{escape(subdomain)}</span>' if subdomain else ""
    rationale_html = f'<div class="rationale">{escape(rationale)}</div>' if rationale else ''

    return (
        f'<tr>'
        f'<td class="title-cell">'
        f'  <a href="{escape(report_link)}">{escape(title)}</a>{policy_badge}'
        f'  <div class="doi">{escape(doi)}</div>'
        f'  {rationale_html}'
        f'</td>'
        f'<td>{field_tag} {sub_tag}</td>'
        f'<td class="persona">{escape(persona)}</td>'
        f'</tr>'
    )


def _section(verdict: str, entries: list[dict]) -> str:
    if not entries:
        return ""
    label, color = VERDICT_META[verdict]
    rows = "\n".join(_row(e) for e in entries)
    return (
        f'<section>'
        f'<h2 style="color:{color}">{label} <span class="count">— {len(entries)}</span></h2>'
        f'<table>'
        f'  <thead><tr><th>Paper</th><th>Field</th><th>Persona</th></tr></thead>'
        f'  <tbody>{rows}</tbody>'
        f'</table>'
        f'</section>'
    )


def build_html(entries: list[dict]) -> str:
    # Group by verdict
    buckets: dict[str, list[dict]] = {v: [] for v in VERDICT_ORDER}
    for e in entries:
        buckets[e["verdict"]].append(e)

    # Sort each bucket by DOI (deterministic ordering)
    for v in buckets:
        buckets[v].sort(key=lambda x: x["doi_label"])

    header_bits = []
    for v in VERDICT_ORDER:
        if buckets[v]:
            label, _ = VERDICT_META[v]
            header_bits.append(f"{label}: {len(buckets[v])}")
    header = " · ".join(header_bits) if header_bits else "No papers processed yet."

    sections = "\n".join(_section(v, buckets[v]) for v in VERDICT_ORDER)

    css = """
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }
      h1 { margin-bottom: 0.25rem; }
      .summary { color: #555; margin-bottom: 2rem; }
      section { margin: 2rem 0; }
      h2 { border-bottom: 2px solid currentColor; padding-bottom: 0.35rem; margin-bottom: 0.75rem; }
      .count { color: #888; font-weight: normal; font-size: 0.85em; }
      table { width: 100%; border-collapse: collapse; }
      th, td { text-align: left; padding: 0.55rem 0.5rem; vertical-align: top; border-bottom: 1px solid #eee; }
      th { font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.04em; color: #666; }
      .title-cell a { color: #0a4f9f; text-decoration: none; font-weight: 500; }
      .title-cell a:hover { text-decoration: underline; }
      .doi { font-family: ui-monospace, monospace; color: #888; font-size: 0.8em; margin-top: 2px; }
      .rationale { color: #444; font-size: 0.9em; margin-top: 4px; font-style: italic; }
      .tag { display: inline-block; padding: 2px 8px; background: #eef; color: #335; border-radius: 10px; font-size: 0.75em; font-weight: 600; }
      .tag.muted { background: #f0f0f0; color: #666; font-weight: 400; }
      .persona { font-size: 0.85em; color: #555; }
    """

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PeerLens editorial dashboard</title>
<style>{css}</style>
</head><body>
<h1>PeerLens editorial dashboard</h1>
<div class="summary">{escape(header)}</div>
{sections}
</body></html>
"""


def main() -> None:
    entries = load_all_verdicts()
    html = build_html(entries)
    out = get_path("processed_data") / "editorial_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard written: {out}")


if __name__ == "__main__":
    main()
