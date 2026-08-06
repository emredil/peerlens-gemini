"""Novelty / Literature Agent (Gemini port).

Retrieval + Gemini pro-latest reasoning. Produces missed_citation /
overreach / staleness / originality findings.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient
from peerlens.agents.corpus_index import find_similar_to_doi, SimilarPaper
from peerlens.agents.semantic_scholar import (
    get_paper as ss_get_paper,
    get_recent_related as ss_get_recent_related,
    ScholarPaper,
)


@dataclass
class RelatedWork:
    title: str
    year: Optional[int]
    authors: list[str]
    venue: str
    source: str
    similarity: float
    citation_count: int = 0
    external_id: str = ""
    cited_by_paper: bool = False


@dataclass
class NoveltyFinding:
    kind: str
    severity: str
    location: str
    evidence: str
    suggested_fix: str = ""
    references: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class NoveltyReport:
    doi_label: str
    narrative: str = ""
    verdict: str = ""  # "clearly novel" | "incremental" | "overlaps with prior work" 
    n_related_retrieved: int = 0
    n_scholar_matches: int = 0
    n_corpus_matches: int = 0
    n_uncited_but_close: int = 0
    n_findings: int = 0
    findings: list[NoveltyFinding] = field(default_factory=list)
    reference_year_median: Optional[int] = None
    reference_year_max: Optional[int] = None
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _titles_match(a: str, b: str) -> bool:
    na, nb = _norm_title(a).split(), _norm_title(b).split()
    if not na or not nb:
        return False
    prefix_a = " ".join(na[:6])
    prefix_b = " ".join(nb[:6])
    return prefix_a == prefix_b or prefix_a in " ".join(nb) or prefix_b in " ".join(na)


def _corpus_to_related(sim: SimilarPaper) -> RelatedWork:
    return RelatedWork(
        title=sim.title,
        year=None,
        authors=[],
        venue="SciPost Physics",
        source="corpus",
        similarity=sim.similarity,
    )


def _scholar_to_related(sp: ScholarPaper) -> RelatedWork:
    ext_id = ""
    if sp.external_ids:
        for key in ("DOI", "ArXiv", "MAG", "CorpusId"):
            if key in sp.external_ids and sp.external_ids[key]:
                ext_id = f"{key}:{sp.external_ids[key]}"
                break
    return RelatedWork(
        title=sp.title,
        year=sp.year,
        authors=sp.authors[:5],
        venue=sp.venue,
        source="scholar",
        similarity=0.0,
        citation_count=sp.citation_count,
        external_id=ext_id,
    )


def gather_related(doi: str, k_corpus: int = 8, k_scholar: int = 15) -> list[RelatedWork]:
    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not parsed_path.exists():
        return []

    corpus_hits = find_similar_to_doi(doi, n=k_corpus)
    result: list[RelatedWork] = [_corpus_to_related(s) for s in corpus_hits]

    ss_paper = ss_get_paper(f"DOI:10.21468/{doi}")
    ss_related: list[ScholarPaper] = []
    if ss_paper and ss_paper.paper_id:
        ss_related = ss_get_recent_related(ss_paper.paper_id, limit=k_scholar, year_from=2022)

    for sp in ss_related:
        if not sp.title.strip():
            continue
        matched = None
        for r in result:
            if _titles_match(sp.title, r.title):
                matched = r
                break
        if matched is not None:
            matched.source = "both"
            if not matched.year:
                matched.year = sp.year
            if not matched.authors:
                matched.authors = sp.authors[:5]
            if matched.citation_count == 0:
                matched.citation_count = sp.citation_count
            if not matched.external_id and sp.external_ids:
                matched.external_id = next(
                    (f"{k}:{v}" for k, v in sp.external_ids.items() if v),
                    "",
                )
        else:
            result.append(_scholar_to_related(sp))
    return result


def _extract_year(ref: str) -> Optional[int]:
    # Strip arXiv identifiers (e.g. "arXiv:2093.05231" or "1903.03686")
    # before scanning for a year, because arXiv IDs share the same
    # 4-digit prefix pattern as years and would otherwise be treated
    # as reference years.
    cleaned = re.sub(r"arXiv[:\s]*\d{4}\.\d{4,5}", " ", ref, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d{4}\.\d{4,5}\b", " ", cleaned)
    # Match a 4-digit year that is not followed by a dot+digits (which
    # would indicate an arXiv-style identifier we did not catch above).
    m = re.search(r"\b(19|20)\d{2}\b(?!\.\d)", cleaned)
    return int(m.group(0)) if m else None


def _reference_bag(ref: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", ref.lower())


def _is_cited(work: RelatedWork, ref_bags: list[str]) -> bool:
    title_words = [w for w in _norm_title(work.title).split() if len(w) > 3][:4]
    if not title_words:
        return False
    title_phrase = " ".join(title_words)
    for bag in ref_bags:
        if title_phrase in bag:
            return True
        for a in work.authors:
            last = a.split()[-1].lower() if a.strip() else ""
            if last and last in bag:
                hits = sum(1 for w in title_words if w in bag)
                if hits >= 2:
                    return True
    return False


def annotate_citations(paper_refs: list[str], related: list[RelatedWork]) -> None:
    ref_bags = [_reference_bag(r) for r in paper_refs]
    for r in related:
        r.cited_by_paper = _is_cited(r, ref_bags)


def reference_year_stats(paper_refs: list[str]) -> tuple[Optional[int], Optional[int]]:
    years = [_extract_year(r) for r in paper_refs]
    from datetime import date
    upper = date.today().year + 1
    years = [y for y in years if y is not None and 1900 <= y <= upper]
    if not years:
        return None, None
    years.sort()
    median = years[len(years)//2]
    return median, max(years)


NOVELTY_REVIEWER_PROMPT = """You are reviewing a scientific manuscript for a journal editor.
Your task is to assess its novelty relative to the existing literature.

You are given:
- The manuscript itself (attached as PDF)
- A list of the most semantically similar papers found in our corpus,
  with their titles, abstracts, and years

Write a novelty assessment as a working reviewer would write it, in prose,
addressed to the editor. Cover the following in a natural flow:

1. What is the manuscript's central claim of novelty? Read the abstract
   and introduction carefully; identify what the authors say is new.

2. How does this claim hold up against the similar papers listed?
   For each closely related paper (say the top three or four), briefly
   compare: what does the current manuscript do that this related paper
   does not, or what does the related paper already do that the current
   manuscript re-invents.

3. Is the current manuscript missing citations to relevant prior work?
   Check whether papers in the similar-papers list appear in the
   manuscript's reference list. If a highly similar paper (top 3) is
   not cited, flag it as a gap.

4. Overall judgment: is the manuscript's novelty claim substantiated?
   Three levels: clearly novel (a real contribution to the field),
   incremental (a modest advance over existing work, still publishable),
   or overlaps with prior work (the claimed contribution is already
   present in the literature).

Keep the assessment under 200 words. Do not use bullet points
or numbered lists in your output. Do not include any metadata such as
confidence scores, agent names, or JSON. Write the assessment as you
would in a real referee report.

At the end, on a separate line, write your overall judgment exactly in
this format (no extra words):
Novelty verdict: clearly novel
Novelty verdict: incremental
Novelty verdict: overlaps with prior work
Pick one of the three.

Use LaTeX notation for any mathematical symbols and equations in your prose. Write `$\\varepsilon$` not `ε`, `$\\mu$` not `µ`, `$\\pi$` not `π`. Every mathematical symbol must be wrapped in `$...$` so the report renders cleanly in markdown with math support."""


NOVELTY_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["missed_citation", "overreach", "staleness", "originality", "info"],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "warning", "info"],
                    },
                    "location": {"type": "string"},
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "references": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["kind", "severity", "evidence", "confidence"],
            },
        },
    },
    "required": ["findings"],
}


NOVELTY_SYSTEM = """You are a physics referee evaluating a manuscript's novelty and
literature coverage. You do NOT have the full paper — only the title,
abstract, section headings, its own reference list, and a curated list of
recent related work (from the arXiv-linked scholarly graph and a domain
corpus of SciPost Physics papers).

Your findings should be actionable for a human editor and referee, NOT a
list of accusations. Prefer specificity and understatement.

CATEGORIES:

- missed_citation: A related paper is clearly relevant to the manuscript's
  claims but is NOT in the paper's reference list. Be conservative: only
  flag if the missed paper is either (a) a well-cited precursor (>=10
  citations, published >=1 year before this manuscript), or (b) a direct
  antecedent from the same authors or same subfield. Do NOT flag brand-new
  papers the author could not reasonably have known.

- overreach: The abstract or title makes a novelty claim ("first",
  "novel", "for the first time", "unprecedented") that the retrieved
  related work already achieves in a similar setting. Cite the specific
  related paper and quote the claim.

- staleness: The paper's reference list is dominated by pre-2020 citations
  while the field has moved substantially since then. Flag only when a
  clear gap of >=3 years between the median reference and the publication
  year is present, AND at least one recent (last 2 years) key paper is
  missing.

- originality: A positive note when the paper genuinely does something the
  related work does not — no severity=critical here. Use severity=info.

- info: Any other observation that is descriptive rather than prescriptive.

SEVERITY:
- critical: The paper's central novelty claim is likely wrong.
- warning: A specific, actionable gap the referee should ask about.
- info: Descriptive; no action needed.

Produce at most 5-8 findings. Zero findings is a valid answer if the
literature coverage is thorough. Do not fabricate findings to fill a quota."""


def _format_related(related: list[RelatedWork]) -> str:
    lines = []
    for i, r in enumerate(related):
        cited = "CITED" if r.cited_by_paper else "not cited"
        yr = str(r.year) if r.year else "?"
        auth = ", ".join(r.authors[:3]) if r.authors else "?"
        line = (
            f"  [{i}] [{cited}] [{r.source}] "
            f"({yr}) cites={r.citation_count} sim={r.similarity:.2f}\n"
            f"       Title: {r.title}\n"
            f"       Authors: {auth}\n"
            f"       Venue: {r.venue}"
        )
        if r.external_id:
            line += f"\n       ID: {r.external_id}"
        lines.append(line)
    return "\n".join(lines)


def _format_paper_context(parsed: dict, refs: list[str],
                          med_yr: Optional[int], max_yr: Optional[int]) -> str:
    section_titles = ", ".join(
        f"{s['number']} {s['title']}"
        for s in parsed.get("sections", [])[:12]
    )
    refs_first_8 = "\n".join(f"  {r[:200]}" for r in refs[:8])
    return f"""TITLE:
{parsed.get('title', '')}

ABSTRACT:
{parsed.get('abstract', '')}

SECTION HEADINGS:
{section_titles}

REFERENCE STATS:
  total references: {len(refs)}
  median reference year: {med_yr}
  most recent reference year: {max_yr}

FIRST 8 REFERENCES (as printed):
{refs_first_8}"""


def analyze_novelty(doi: str, client: Optional[GeminiClient] = None, uploaded_file: Any = None) -> NoveltyReport:
    if client is None:
        client = GeminiClient()

    report = NoveltyReport(doi_label=doi)

    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not parsed_path.exists():
        return report
    parsed = json.loads(parsed_path.read_text())
    refs = parsed.get("references", [])

    med_yr, max_yr = reference_year_stats(refs)
    report.reference_year_median = med_yr
    report.reference_year_max = max_yr

    related = gather_related(doi)
    annotate_citations(refs, related)

    report.n_related_retrieved = len(related)
    report.n_scholar_matches = sum(1 for r in related if r.source in ("scholar", "both"))
    report.n_corpus_matches = sum(1 for r in related if r.source in ("corpus", "both"))
    report.n_uncited_but_close = sum(1 for r in related if not r.cited_by_paper)

    if not related:
        report.findings.append(NoveltyFinding(
            kind="info",
            severity="info",
            location="Retrieval",
            evidence="No related work could be retrieved. Automated novelty check is not applicable.",
            confidence=1.0,
        ))
        report.n_findings = len(report.findings)
        return report

    user = f"""PAPER CONTEXT:

{_format_paper_context(parsed, refs, med_yr, max_yr)}


RETRIEVED RELATED WORK ({len(related)} items):

{_format_related(related)}


Assess this manuscript's novelty and literature coverage using the
record_novelty_findings tool. Focus on the four categories described in
the system prompt. Zero findings is acceptable if coverage is thorough."""

    pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
    # Prefer uploaded_file (cached on Gemini's side) over inline pdf_bytes.
    pdf_bytes = None
    if uploaded_file is None:
        pdf_bytes = pdf_path.read_bytes() if pdf_path.exists() else None

    user_prose = (
        "SIMILAR PAPERS FOUND IN OUR CORPUS "
        f"({len(related)} items, ordered by semantic similarity):\n\n"
        + _format_related(related)
        + "\n\nReference-year statistics of the manuscript: "
        + (f"median {med_yr}, most recent {max_yr}." if med_yr else "unavailable.")
    )

    model_name = CONFIG["gemini"]["model_specialist"]
    text, in_tok, out_tok = client.plain_text(
        model=model_name,
        system=NOVELTY_REVIEWER_PROMPT,
        user=user_prose,
        max_output_tokens=8000,
        temperature=0.3,
        pdf_bytes=pdf_bytes,
        uploaded_file=uploaded_file,
    )

    narrative_lines = []
    verdict = ""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("novelty verdict:"):
            verdict = stripped.split(":", 1)[1].strip().lower()
        else:
            narrative_lines.append(line)
    report.narrative = "\n".join(narrative_lines).strip()
    report.verdict = verdict

    report.n_findings = 0
    report.model = model_name
    report.input_tokens = in_tok
    report.output_tokens = out_tok
    return report


def save_report(report: NoveltyReport, out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "novelty_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
