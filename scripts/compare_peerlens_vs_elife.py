"""Compare PeerLens verdicts with eLife Assessment tags."""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from peerlens.settings import get_path


def load_msids(path: Path) -> list[str]:
    ids = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ids.append(ln.replace("elife_", ""))
    return ids


def peerlens_data(doi_label: str) -> dict:
    p = get_path("processed_data") / "domain_reports" / f"{doi_label}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def peerlens_recommendation(doi_label: str) -> str:
    p = get_path("processed_data") / "referee_sections" / f"{doi_label}_full.md"
    if not p.exists():
        return ""
    m = re.search(r"^\*\*(Accept|Minor revision|Major revision|Reject)\*\*",
                  p.read_text(), re.M | re.I)
    return m.group(1).lower() if m else ""


def elife_assessment(msid: str) -> dict:
    p = get_path("processed_data") / "elife_assessments" / f"{msid}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def shorten(text: str, limit: int = 180) -> str:
    text = (text or "").replace("|", "/").replace("\n", " ").strip()
    if not text:
        return ""
    m = re.search(r"[.!?](\s|$)", text)
    if m and m.start() + 1 <= limit:
        return text[:m.start() + 1]
    return text[:limit].rstrip() + ("..." if len(text) > limit else "")


def main():
    msids = load_msids(Path(sys.argv[1]))
    rows = []
    verdict_counter: Counter = Counter()
    sig_counter: Counter = Counter()
    str_counter: Counter = Counter()

    for msid in msids:
        doi_label = f"elife_{msid}"
        pl = peerlens_data(doi_label)
        pl_verdict = peerlens_recommendation(doi_label) or (pl.get("verdict") or "unknown").lower()
        pl_rationale = pl.get("verdict_rationale", "")
        ea = elife_assessment(msid)
        sig = " / ".join(ea.get("significance", []))
        stre = " / ".join(ea.get("strength", []))

        verdict_counter[pl_verdict] += 1
        for s in ea.get("significance", []):
            sig_counter[s] += 1
        for s in ea.get("strength", []):
            str_counter[s] += 1

        rows.append({
            "msid": msid,
            "title": (ea.get("title_full") or pl.get("title") or "")[:80],
            "peerlens_verdict": pl_verdict,
            "peerlens_rationale": pl_rationale,
            "significance": sig or "n/a",
            "strength": stre or "n/a",
        })

    out = get_path("processed_data") / "comparison_peerlens_vs_elife.md"
    lines = [
        "# PeerLens vs. eLife Assessment tags",
        "",
        f"Corpus: {len(msids)} eLife reviewed preprints.",
        "",
        "eLife editors do not issue single-word verdicts; they publish an",
        "'eLife Assessment' with two tags — *significance* (landmark, fundamental,",
        "important, valuable, useful) and *strength* (exceptional, compelling,",
        "convincing, solid, incomplete, inadequate). We present these alongside",
        "PeerLens' own accept/minor/major/reject verdict without imposing a",
        "one-to-one mapping.",
        "",
        "## PeerLens verdict distribution",
        "",
        "| Verdict | Count |",
        "|---|---|",
    ]
    for v, n in verdict_counter.most_common():
        lines.append(f"| {v} | {n} |")
    lines += [
        "",
        "## eLife significance distribution",
        "",
        "| significance | Count |",
        "|---|---|",
    ]
    for v, n in sig_counter.most_common():
        lines.append(f"| {v} | {n} |")
    lines += [
        "",
        "## eLife strength distribution",
        "",
        "| strength | Count |",
        "|---|---|",
    ]
    for v, n in str_counter.most_common():
        lines.append(f"| {v} | {n} |")
    lines += [
        "",
        "## Per-paper detail",
        "",
        "| Paper | PeerLens verdict | eLife significance | eLife strength | PeerLens rationale |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| elife_{r['msid']} | {r['peerlens_verdict']} | {r['significance']} | "
            f"{r['strength']} | {shorten(r['peerlens_rationale'], 200)} |"
        )

    out.write_text("\n".join(lines))
    print(f"\nWritten: {out}\n")
    print("PeerLens verdicts:", dict(verdict_counter))
    print("eLife significance:", dict(sig_counter))
    print("eLife strength:", dict(str_counter))


if __name__ == "__main__":
    main()
