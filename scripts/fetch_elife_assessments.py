"""Fetch elifeAssessment metadata for a list of eLife article IDs and save
each to data/processed/elife_assessments/{id}.json.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from peerlens.settings import get_path


API_URL = "https://api.elifesciences.org/reviewed-preprints/{msid}"
HEADERS = {
    "Accept": "application/vnd.elife.reviewed-preprint+json; version=1",
    "User-Agent": "PeerLens/0.0.1 (research prototype)",
}


def _split_reviewers(html: str) -> list[dict]:
    """Split indexContent HTML into per-reviewer blocks.

    eLife wraps each reviewer's Public Review in a `<strong>Reviewer #N
    (Public Review):</strong>` header. We slice on those anchors and keep
    only the first occurrence per reviewer (later blocks are usually the
    "recommendations to authors" section which is duplicated across
    revisions).
    """
    import re
    from html import unescape
    if not html:
        return []
    # Find header positions
    starts = [(m.start(), m.group(1))
              for m in re.finditer(r"Reviewer #(\d+) \(Public Review\)", html)]
    if not starts:
        return []
    starts.append((len(html), None))
    seen: set[str] = set()
    reviewers: list[dict] = []
    for i, (pos, num) in enumerate(starts[:-1]):
        if num in seen:
            continue
        seen.add(num)
        block = html[pos:starts[i + 1][0]]
        # Strip tags; collapse whitespace
        text = re.sub(r"<[^>]+>", " ", block)
        text = unescape(re.sub(r"\s+", " ", text)).strip()
        reviewers.append({"num": num, "text": text})
    return reviewers


def fetch_assessment(msid: str, client: httpx.Client) -> dict:
    r = client.get(API_URL.format(msid=msid))
    r.raise_for_status()
    d = r.json()
    ea = d.get("elifeAssessment", {}) or {}
    prose = "\n\n".join(
        c.get("text", "")
        for c in ea.get("content", [])
        if c.get("type") == "paragraph"
    ).strip()
    reviewers = _split_reviewers(d.get("indexContent", "") or "")
    return {
        "msid": msid,
        "doi": d.get("doi"),
        "title": d.get("title"),
        "significance": ea.get("significance", []),
        "strength": ea.get("strength", []),
        "assessment_prose": prose,
        "reviewers": reviewers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True,
                    help="eLife manuscript IDs, e.g. 96032 107671 ...")
    args = ap.parse_args()

    out_dir = get_path("processed_data") / "elife_assessments"
    out_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers=HEADERS, timeout=30) as client:
        for msid in args.ids:
            try:
                data = fetch_assessment(msid, client)
                (out_dir / f"{msid}.json").write_text(
                    json.dumps(data, indent=2, ensure_ascii=False)
                )
                print(f"OK   elife_{msid}  "
                      f"significance={data['significance']}  "
                      f"strength={data['strength']}")
            except Exception as e:
                print(f"FAIL elife_{msid}  {type(e).__name__}: {str(e)[:120]}")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
