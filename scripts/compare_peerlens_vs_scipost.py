"""Compare PeerLens verdicts with SciPost referee verdicts (v1-only fair
comparison) and produce a rich markdown report suitable for the manuscript.

Sections in the output:
  1. Aggregate distribution
  2. Per-paper table with PeerLens rationale + SciPost recommendations
  3. Illustrative case studies: one paper per category with the actual
     PeerLens weakness narrative snippet.
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from peerlens.settings import get_path


DOI_LIST_ARG_DEFAULT = "/tmp/scipost_21.txt"

VERDICT_ORDER = ["accept", "minor revision", "major revision", "reject"]

REC_TO_ORDER = {
    "publish": 0,
    "publish (meets expectations and criteria for this journal)": 0,
    "publish in scipost physics": 0,
    "ask for minor revision": 1,
    "ask for major revision": 2,
    "reject": 3,
}

PEERLENS_TO_ORDER = {
    "accept": 0,
    "minor revision": 1,
    "major revision": 2,
    "reject": 3,
}


def load_dois(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def peerlens_data(doi: str) -> dict:
    p = get_path("processed_data") / "domain_reports" / f"{doi}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def peerlens_recommendation_from_report(doi: str) -> str:
    p = get_path("processed_data") / "referee_sections" / f"{doi}_full.md"
    if not p.exists():
        return ""
    m = re.search(r"^\*\*(Accept|Minor revision|Major revision|Reject)\*\*",
                  p.read_text(), re.M | re.I)
    return m.group(1).lower() if m else ""


def scipost_v1_reviews(doi: str) -> list[dict]:
    """Only v1 reports, for fair comparison against PeerLens v1 output."""
    p = get_path("raw_data") / "reports" / f"{doi}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    versions = d.get("versions", {})
    if "1" not in versions:
        return []
    return [{
        "recommendation": (r.get("recommendation") or "").strip().lower(),
        "reviewer": r.get("reviewer_name") or "anonymous",
    } for r in versions["1"]]


def scipost_summary_verdict(reviews: list[dict]) -> str:
    if not reviews:
        return "no_reviews"
    scores = []
    for r in reviews:
        rec = r["recommendation"]
        for key, order in REC_TO_ORDER.items():
            if key in rec:
                scores.append(order)
                break
    if not scores:
        return "no_recommendation"
    scores.sort()
    return VERDICT_ORDER[scores[len(scores) // 2]]


def categorize(peerlens_v: str, scipost_v: str) -> str:
    if scipost_v.startswith("no"):
        return "no_ground_truth"
    if peerlens_v not in PEERLENS_TO_ORDER or scipost_v not in PEERLENS_TO_ORDER:
        return "unknown"
    p = PEERLENS_TO_ORDER[peerlens_v]
    s = PEERLENS_TO_ORDER[scipost_v]
    if p == s:
        return "agree"
    if p > s:
        return "peerlens_stricter"
    return "peerlens_lighter"


def shorten(text: str, limit: int = 140) -> str:
    """Trim to first sentence or `limit` chars; strip markdown pipe chars."""
    text = (text or "").replace("|", "/").replace("\n", " ").strip()
    if not text:
        return ""
    m = re.search(r"[.!?](\s|$)", text)
    if m and m.start() + 1 <= limit:
        return text[:m.start() + 1]
    return text[:limit].rstrip() + ("..." if len(text) > limit else "")


def main():
    doi_list_path = Path(sys.argv[1] if len(sys.argv) > 1 else DOI_LIST_ARG_DEFAULT)
    dois = load_dois(doi_list_path)
    rows = []
    counter: Counter = Counter()

    for doi in dois:
        pl = peerlens_data(doi)
        pl_rec = peerlens_recommendation_from_report(doi)
        pl_verdict = pl_rec or (pl.get("verdict") or "unknown").lower()
        pl_rationale = pl.get("verdict_rationale", "")
        pl_weaknesses = pl.get("weaknesses_narrative", "")

        reviews = scipost_v1_reviews(doi)
        sp_verdict = scipost_summary_verdict(reviews)
        cat = categorize(pl_verdict, sp_verdict)
        counter[cat] += 1

        rows.append({
            "doi": doi,
            "peerlens_verdict": pl_verdict,
            "peerlens_rationale": pl_rationale,
            "peerlens_weaknesses": pl_weaknesses,
            "scipost_verdict": sp_verdict,
            "scipost_n_reports": len(reviews),
            "scipost_recs": [r["recommendation"] for r in reviews],
            "category": cat,
        })

    out_path = get_path("processed_data") / "comparison_peerlens_vs_scipost.md"

    lines = [
        "# PeerLens vs. SciPost Physics referees (v1-only fair comparison)",
        "",
        f"Corpus: {len(dois)} SciPost Physics papers.",
        "",
        "Comparison uses each paper's *first-round* reviewer reports, because",
        "PeerLens analyses the initially-submitted PDF (v1). Later versions",
        "typically carry 'publish' verdicts after the author has addressed",
        "round-one concerns, which would understate PeerLens' agreement.",
        "",
        "## 1. Aggregate distribution",
        "",
        "| Category | Count | % of ground-truth papers |",
        "|---|---|---|",
    ]
    n_gt = sum(v for k, v in counter.items() if k != "no_ground_truth")
    for cat in ("agree", "peerlens_stricter", "peerlens_lighter", "no_ground_truth", "unknown"):
        n = counter.get(cat, 0)
        if n == 0:
            continue
        pct = f"{100*n/n_gt:.0f}%" if cat != "no_ground_truth" and n_gt else ""
        lines.append(f"| {cat} | {n} | {pct} |")

    lines += ["", "## 2. Per-paper detail", "",
              "| DOI | PeerLens verdict | PeerLens rationale | SciPost verdict | Reports (v1) | Category |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['doi']} | {r['peerlens_verdict']} | {shorten(r['peerlens_rationale'], 180)} | "
            f"{r['scipost_verdict']} | {r['scipost_n_reports']} | {r['category']} |"
        )

    # Case studies: one example per category
    lines += ["", "## 3. Case studies"]
    for cat in ("peerlens_stricter", "agree", "peerlens_lighter"):
        examples = [r for r in rows if r["category"] == cat]
        if not examples:
            continue
        ex = examples[0]
        lines += [
            "",
            f"### {cat.replace('_', ' ').title()} — {ex['doi']}",
            "",
            f"- **PeerLens:** {ex['peerlens_verdict']}",
            f"- **SciPost referees (v1):** {ex['scipost_verdict']}  "
            f"({ex['scipost_n_reports']} report(s): {'; '.join(ex['scipost_recs']) or 'n/a'})",
            "",
            "**PeerLens rationale:**",
            "",
            f"> {ex['peerlens_rationale'].strip() or '(none)'}",
            "",
            "**PeerLens weaknesses (excerpt):**",
            "",
            f"> {shorten(ex['peerlens_weaknesses'], 700)}",
        ]

    out_path.write_text("\n".join(lines))
    print(f"\nWritten: {out_path}\n")
    print("Distribution:")
    for cat, n in counter.most_common():
        print(f"  {cat:22s}  {n}")


if __name__ == "__main__":
    main()
