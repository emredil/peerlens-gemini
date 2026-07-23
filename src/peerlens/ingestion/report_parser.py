"""Enhanced parser for SciPost referee reports.

Improvements over the initial parser:
- Score extraction uses SciPost's controlled vocabulary (whitelist), so
  it no longer captures free-text from Strengths sections.
- Report body is separated from any embedded author reply.
- Structured optional sections (Strengths, Weaknesses, Requested changes)
  are extracted into their own fields.
- Date is normalized to YYYY-MM-DD.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup


SCORE_FIELDS = ["validity", "significance", "originality", "clarity", "formatting", "grammar"]

# SciPost score dropdown values (controlled vocabulary)
SCORE_VALUES_RE = r"(top|high|good|ok|low|poor|acceptable|excellent|reasonable|perfect|-)"


@dataclass
class PeerReport:
    """A single referee report parsed from a SciPost submission page."""
    submission_id: str
    version: int
    report_number: int
    referee_identity: str
    invited: bool
    date: str
    strengths: str
    weaknesses: str
    body: str
    requested_changes: str
    recommendation: str
    scores: dict
    author_reply: Optional[dict]  # {author_name, date, comment_id, body}
    raw_text: str


def _normalize_date(s: str) -> str:
    """Turn '2026-2-1' into '2026-02-01'."""
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        pass
    parts = s.split("-")
    if len(parts) == 3:
        try:
            return f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        except ValueError:
            pass
    return s


def _extract_header(text: str) -> tuple[str, bool, str, int]:
    """Return (identity, invited, date, end_position_of_header)."""
    m = re.search(
        r"#?(\d+)\s+by\s+(.+?)\s+on\s+(\d{4}-\d{1,2}-\d{1,2})\s*\((Invited|Contributed)\s+Report\)",
        text,
    )
    if m:
        return m.group(2).strip(), m.group(4) == "Invited", _normalize_date(m.group(3)), m.end()
    return "Unknown", False, "", 0


def _split_report_and_reply(text: str) -> tuple[str, Optional[dict]]:
    """Separate the referee report portion from an embedded author reply, if any.

    An author reply block is anchored by:  Author: <name> on YYYY-MM-DD [id NNNN]
    """
    m = re.search(
        r"Author:\s+(.+?)\s+on\s+(\d{4}-\d{1,2}-\d{1,2})\s+\[id\s+(\d+)\]",
        text,
    )
    if not m:
        return text, None

    report_text = text[:m.start()].strip()
    reply = {
        "author_name": m.group(1).strip(),
        "date": _normalize_date(m.group(2)),
        "comment_id": f"comment_id{m.group(3)}",
        "body": text[m.end():].strip(),
    }
    return report_text, reply


def _extract_scores(text: str) -> dict:
    """Whitelist-based score extraction."""
    scores = {f: None for f in SCORE_FIELDS}
    for f in SCORE_FIELDS:
        m = re.search(rf"\b{f}\s*:\s*{SCORE_VALUES_RE}\b", text, re.IGNORECASE)
        if m:
            scores[f] = m.group(1).lower()
    return scores


def _extract_recommendation(text: str) -> str:
    """Text between 'Recommendation' keyword and the first score field."""
    m = re.search(
        r"Recommendation\s+(.+?)\s+(?:validity|significance)\s*:\s*",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def _extract_optional_section(text: str, section: str, next_sections: list[str]) -> str:
    """Extract text between a section header and the next known header."""
    m = re.search(rf"\b{re.escape(section)}\b\s+", text)
    if not m:
        return ""
    tail = text[m.end():]

    earliest = len(tail)
    for ns in next_sections:
        nm = re.search(rf"\s+\b{re.escape(ns)}\b\s+", tail)
        if nm and nm.start() < earliest:
            earliest = nm.start()
    return tail[:earliest].strip()


def _clean_preamble(text: str) -> str:
    """Remove Cite as: and AI Disclosure preambles that precede the actual sections."""
    text = re.sub(
        r"Cite as:.+?(?=Strengths|Weaknesses|AI Disclosure|Report\s+[A-Z]|Recommendation)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"AI Disclosure.+?(?=Strengths|Weaknesses|Report\s+[A-Z]|Recommendation)",
        "",
        text,
        flags=re.DOTALL,
    )
    return text


def _extract_body(text: str) -> str:
    """Extract main report body between the 'Report' section header and 'Recommendation'."""
    # Recommendation anchors the end
    rec_m = re.search(r"\bRecommendation\b", text)
    body_end = rec_m.start() if rec_m else len(text)
    search_area = text[:body_end]

    # Find the last standalone 'Report' before Recommendation
    report_positions = list(re.finditer(r"\bReport\b", search_area))
    if not report_positions:
        return search_area.strip()

    body_start = report_positions[-1].end()
    body_text = text[body_start:body_end]

    # If 'Requested changes' appears inside body, trim it
    rq_m = re.search(r"\bRequested changes\b", body_text)
    if rq_m:
        body_text = body_text[:rq_m.start()]

    return body_text.strip()


def parse_reports_from_html(html: str, submission_id: str, version: int) -> list[PeerReport]:
    """Extract all referee reports from a SciPost submission HTML page."""
    soup = BeautifulSoup(html, "lxml")
    reports = []

    for div in soup.find_all("div", class_="report"):
        report_id = div.get("id", "")
        m = re.match(r"report_(\d+)", report_id)
        if not m:
            continue
        report_number = int(m.group(1))

        raw = div.get_text(separator=" ", strip=True)
        raw = re.sub(r"\s+", " ", raw)

        # 1. Separate report from author reply
        report_text, reply = _split_report_and_reply(raw)

        # 2. Header (identity, invited flag, date)
        identity, invited, date, header_end = _extract_header(report_text)

        # 3. Strip preambles
        content = report_text[header_end:] if header_end else report_text
        content = _clean_preamble(content)

        # 4. Structured optional sections
        strengths = _extract_optional_section(
            content, "Strengths",
            ["Weaknesses", "Report", "Requested changes", "Recommendation"],
        )
        weaknesses = _extract_optional_section(
            content, "Weaknesses",
            ["Report", "Requested changes", "Recommendation"],
        )
        requested = _extract_optional_section(
            content, "Requested changes",
            ["Recommendation"],
        )

        body = _extract_body(content)
        recommendation = _extract_recommendation(content)
        scores = _extract_scores(content)

        reports.append(PeerReport(
            submission_id=submission_id,
            version=version,
            report_number=report_number,
            referee_identity=identity,
            invited=invited,
            date=date,
            strengths=strengths,
            weaknesses=weaknesses,
            body=body,
            requested_changes=requested,
            recommendation=recommendation,
            scores=scores,
            author_reply=reply,
            raw_text=raw,
        ))

    return reports
