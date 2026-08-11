"""F1000Research ingestion client.

Fetches paper metadata (title, authors, verdict, reviewer reports) from the
public F1000Research Solr API and downloads the underlying PDF via the
extapi endpoint.

Reviewer reports are our ground truth for maths/chemistry testing:
each report carries an F1000 recommendation ("approve" /
"approve-with-reservations" / "reject") which we aggregate into an overall
paper verdict using F1000's own indexing rule.

Public API base:       https://f1000research.com/extapi
Search endpoint:       /search?q=<Solr query>
Article XML endpoint:  /article/xml?doi=<doi>
Article PDF endpoint:  /article/pdf?doi=<doi>
Docs:                  https://f1000research-files.f1000.com/resources/F1000Research-API-documentation.pdf
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx

from peerlens.settings import get_path


F1000_API_BASE = "https://f1000research.com/extapi"
USER_AGENT = "PeerLens/0.0.1 (research prototype; peer-review evaluation)"
RATE_LIMIT_SLEEP = 1.0  # F1000: 100 req / 60 s → 0.6 s minimum; use 1 s to be safe

# F1000 recommendation values (as they appear in <custom-meta><meta-value>)
REC_APPROVE = "approve"
REC_APPROVE_RESERVATIONS = "approve-with-reservations"
REC_REJECT = "reject"


@dataclass
class F1000ReviewerReport:
    """One reviewer report attached to an F1000 paper."""
    report_id: str
    reviewer_name: str
    reviewer_affiliation: str
    recommendation: str            # approve / approve-with-reservations / reject
    body_text: str                 # full prose


@dataclass
class F1000Paper:
    """Slim metadata record for one F1000Research paper."""
    doi: str                       # e.g. "10.12688/f1000research.181882.1"
    article_id: str                # numeric ID (e.g. "181882")
    version: str                   # "1", "2", ...
    url_slug: str                  # e.g. "15-849"  (may be empty if unknown)
    title: str
    author_line: str
    published: str                 # YYYY-MM-DD
    domain: str                    # e.g. "mathematics" — set from search filter
    verdict: str = ""              # approved / approved_with_reservations / not_approved / awaiting_review
    reviewer_reports: list[F1000ReviewerReport] = field(default_factory=list)

    @property
    def doi_label(self) -> str:
        """Filesystem-safe label used across the pipeline."""
        return f"f1000_{self.article_id}_v{self.version}"


def _headers(accept: str = "*/*") -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": accept}


def _solr_escape(value: str) -> str:
    """Wrap a subject/type value in double quotes for a Solr phrase query."""
    v = value.replace('"', r'\"')
    return f'"{v}"'


def search_dois(
    subject: str,
    article_type: str = "RESEARCH_ARTICLE",
    max_results: int = 20,
    require_reviews: bool = True,
    client: Optional[httpx.Client] = None,
) -> list[str]:
    """Return DOIs matching the Solr query.

    Args:
        subject: F1000 R_SUB value (e.g. "Mathematics", "Chemistry").
        article_type: R_TY filter, defaults to research articles.
        max_results: hard cap on returned DOIs (F1000 max 100/page).
        require_reviews: filter to articles that have at least one reviewer
            report (R_RP:*).
    """
    q_parts = [
        f"R_SUB:{_solr_escape(subject)}",
        f'R_TY:{_solr_escape(article_type)}',
    ]
    if require_reviews:
        q_parts.append("R_RP:*")
    q = " AND ".join(q_parts)

    params = {"q": q, "rows": min(max_results, 100)}
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=30, headers=_headers())
    try:
        resp = client.get(f"{F1000_API_BASE}/search", params=params)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        dois = [el.text.strip() for el in root.findall("doi") if el.text]
    finally:
        if owns_client:
            client.close()
    return dois[:max_results]


def _slug_from_doi(doi: str, client: httpx.Client) -> str:
    """Reverse-lookup the /articles/{vol-num} URL slug from a DOI.

    Not strictly required for pipeline correctness but useful for humans.
    Returns empty string on failure.
    """
    try:
        r = client.get(
            f"https://doi.org/{doi}", follow_redirects=True, timeout=15
        )
        m = re.search(r"/articles/(\d+-\d+)", str(r.url))
        return m.group(1) if m else ""
    except Exception:
        return ""


def _text(el: Optional[ET.Element]) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _parse_reviewer_report(sub: ET.Element) -> F1000ReviewerReport:
    report_id = sub.get("id", "")
    front_stub = sub.find("front-stub")
    name = ""
    aff = ""
    if front_stub is not None:
        contrib = front_stub.find(".//contrib[@contrib-type='author']/name")
        if contrib is not None:
            given = contrib.find("given-names")
            sur = contrib.find("surname")
            name = f"{_text(given)} {_text(sur)}".strip()
        aff_el = front_stub.find(".//aff")
        aff = _text(aff_el)

    recommendation = ""
    for cm in sub.findall(".//custom-meta"):
        name_el = cm.find("meta-name")
        val_el = cm.find("meta-value")
        if (
            name_el is not None
            and val_el is not None
            and name_el.text == "recommendation"
        ):
            recommendation = (val_el.text or "").strip().lower()
            break

    body = _text(sub.find("body"))

    return F1000ReviewerReport(
        report_id=report_id,
        reviewer_name=name,
        reviewer_affiliation=aff,
        recommendation=recommendation,
        body_text=body,
    )


def _aggregate_verdict(reports: list[F1000ReviewerReport]) -> str:
    """Map recommendation counts -> paper-level verdict."""
    recs = [r.recommendation for r in reports if r.recommendation]
    if len(recs) < 2:
        return "awaiting_review"
    reject_count = sum(1 for r in recs if r == REC_REJECT)
    if reject_count > len(recs) / 2:
        return "not_approved"
    if all(r == REC_APPROVE for r in recs):
        return "approved"
    return "approved_with_reservations"


def fetch_paper(
    doi: str,
    domain: str,
    client: Optional[httpx.Client] = None,
) -> F1000Paper:
    """Fetch and parse an F1000 paper's JATS XML into an F1000Paper."""
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=60, headers=_headers())
    try:
        resp = client.get(f"{F1000_API_BASE}/article/xml", params={"doi": doi})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        title_el = root.find(".//front/article-meta/title-group/article-title")
        title = _text(title_el)

        authors = []
        for contrib in root.findall(
            ".//front/article-meta/contrib-group/contrib[@contrib-type='author']"
        ):
            name = contrib.find("name")
            if name is None:
                continue
            given = name.find("given-names")
            sur = name.find("surname")
            n = f"{_text(given)} {_text(sur)}".strip()
            if n:
                authors.append(n)
        author_line = "; ".join(authors)

        pub = root.find(".//front/article-meta/pub-date[@pub-type='epub']")
        if pub is None:
            pub = root.find(".//front/article-meta/pub-date")
        y = _text(pub.find("year")) if pub is not None else ""
        m = _text(pub.find("month")) if pub is not None else ""
        d = _text(pub.find("day")) if pub is not None else ""
        try:
            published = f"{y}-{int(m):02d}-{int(d):02d}" if (y and m and d) else y
        except ValueError:
            published = y

        m2 = re.match(r"10\.12688/f1000research\.(\d+)\.(\d+)", doi)
        article_id = m2.group(1) if m2 else doi.rsplit(".", 2)[-2]
        version = m2.group(2) if m2 else "1"

        slug = _slug_from_doi(doi, client)

        reports = [
            _parse_reviewer_report(sub)
            for sub in root.findall(".//sub-article[@article-type='reviewer-report']")
        ]

        paper = F1000Paper(
            doi=doi,
            article_id=article_id,
            version=version,
            url_slug=slug,
            title=title,
            author_line=author_line,
            published=published,
            domain=domain,
            reviewer_reports=reports,
        )
        paper.verdict = _aggregate_verdict(reports)
        return paper
    finally:
        if owns_client:
            client.close()


def download_pdf(doi: str, out_path: Path, client: Optional[httpx.Client] = None) -> Path:
    """Download the article PDF to out_path."""
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=120, headers=_headers("application/pdf"))
    try:
        resp = client.get(
            f"{F1000_API_BASE}/article/pdf",
            params={"doi": doi},
            follow_redirects=True,
        )
        resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        return out_path
    finally:
        if owns_client:
            client.close()


def save_paper_bundle(
    paper: F1000Paper,
    client: Optional[httpx.Client] = None,
) -> dict[str, Path]:
    """Write the paper's metadata JSON, PDF, and reviewer-reports markdown.

    Paths mirror the eLife module:
      get_path("raw_data") / "pdf" / <doi_label>.pdf
      get_path("processed_data") / "f1000_reviews" / <domain> / metadata / <doi_label>.json
      get_path("processed_data") / "f1000_reviews" / <domain> / reviews / <doi_label>.md
    """
    import json as _json

    pdf_dir = get_path("raw_data") / "pdf"
    meta_dir = get_path("processed_data") / "f1000_reviews" / paper.domain / "metadata"
    review_dir = get_path("processed_data") / "f1000_reviews" / paper.domain / "reviews"

    pdf_path = pdf_dir / f"{paper.doi_label}.pdf"
    meta_path = meta_dir / f"{paper.doi_label}.json"
    review_path = review_dir / f"{paper.doi_label}.md"

    for p in (pdf_dir, meta_dir, review_dir):
        p.mkdir(parents=True, exist_ok=True)

    download_pdf(paper.doi, pdf_path, client=client)

    meta = asdict(paper)
    for r in meta["reviewer_reports"]:
        r["body_text"] = r["body_text"][:400] + ("..." if len(r["body_text"]) > 400 else "")
    meta_path.write_text(_json.dumps(meta, indent=2, ensure_ascii=False))

    lines = [
        f"# {paper.title}",
        "",
        f"- DOI: {paper.doi}",
        f"- URL: https://f1000research.com/articles/{paper.url_slug or '?'}",
        f"- Domain: {paper.domain}",
        f"- Authors: {paper.author_line}",
        f"- Published: {paper.published}",
        f"- Overall verdict (aggregated): **{paper.verdict}**",
        "",
        "---",
        "",
    ]
    for i, r in enumerate(paper.reviewer_reports, 1):
        lines += [
            f"## Reviewer {i}: {r.reviewer_name or '(anonymous)'}",
            f"- Affiliation: {r.reviewer_affiliation or 'N/A'}",
            f"- Report ID: {r.report_id}",
            f"- Recommendation: **{r.recommendation}**",
            "",
            r.body_text,
            "",
            "---",
            "",
        ]
    review_path.write_text("\n".join(lines))

    return {"pdf": pdf_path, "metadata": meta_path, "reviews": review_path}


MODERN_DOI_RE = re.compile(r"^10\.12688/f1000research\.\d+\.\d+$")


def iter_domain_papers(
    subject: str,
    domain_label: str,
    max_papers: int,
) -> list[F1000Paper]:
    """Search + fetch metadata; skip papers still awaiting review.

    Solr search sometimes returns very old DOIs in the legacy
    `<vol>-<num>.v<ver>` format which the current article/xml endpoint
    404s on. We filter those out.
    """
    papers: list[F1000Paper] = []
    with httpx.Client(timeout=60, headers=_headers()) as client:
        # Fetch more DOIs than we need since we filter aggressively.
        candidate_dois = search_dois(
            subject=subject,
            max_results=max(max_papers * 5, 20),
            require_reviews=True,
            client=client,
        )
        skipped_legacy = 0
        for doi in candidate_dois:
            if not MODERN_DOI_RE.match(doi):
                skipped_legacy += 1
                continue
            time.sleep(RATE_LIMIT_SLEEP)
            try:
                paper = fetch_paper(doi, domain=domain_label, client=client)
            except Exception as exc:
                print(f"[warn] {doi}: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            if paper.verdict == "awaiting_review":
                continue
            if not paper.reviewer_reports:
                # Extra safety net: no reviewer reports means we cannot
                # form ground truth for comparison; skip to avoid burning
                # Gemini tokens on it later.
                continue
            papers.append(paper)
            if len(papers) >= max_papers:
                break
        if skipped_legacy:
            print(f"[info] Skipped {skipped_legacy} legacy-format DOI(s)")
    return papers
