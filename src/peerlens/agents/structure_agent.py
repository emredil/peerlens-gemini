"""Structure & Narrative Agent — silence-default.

Detects derivation gaps, hand-wavy proof shortcuts, and orphan references
that human referees typically flag as 'please clarify' or 'please provide
the missing derivation'.

Two stages:
  1. Local pattern scan — locate candidate 'trigger' phrases in section
     bodies (regex; deterministic).
  2. LLM adjudication — for each candidate, decide whether the phrase
     indicates a real problem in context, or is an acceptable rhetorical
     shortcut. Filter is conservative: silence is the default.

Design (post-variance-diagnostic):
  - No soft quota — the previous "at most 5-10 findings" ceiling drove
    inconsistent filtering ([1,7,7] and [2,0,0] patterns on 6x3 probe).
  - Kind enum narrowed to 2 (derivation_gap, orphan_reference).
    handwavy_shortcut absorbed into derivation_gap; info kind removed.
  - Severity enum narrowed to critical/warning (info removed).
  - Post-hoc confidence filter (< 0.5 dropped).
  - Post-hoc dedup on candidate_id.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


def _classification_for(doi: str) -> Optional[dict]:
    path = get_path("processed_data") / "classifications" / f"{doi}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@dataclass
class StructureFinding:
    kind: str          # "derivation_gap" | "orphan_reference"
    severity: str      # "critical" | "warning"
    location: str      # section id or number
    trigger_phrase: str
    context_excerpt: str
    evidence: str      # LLM's assessment
    suggested_fix: str = ""
    confidence: float = 0.0


@dataclass
class StructureReport:
    doi_label: str
    n_candidates_scanned: int = 0
    n_findings: int = 0
    n_critical: int = 0
    n_warning: int = 0
    findings: list[StructureFinding] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Stage 1: pattern scan
# ---------------------------------------------------------------------------

# Phrases that referees repeatedly flag as hand-wavy or gap-indicating.
# Case-insensitive substring match.
TRIGGER_PATTERNS = [
    r"\bit is easy to (?:see|show|verify|check)\b",
    r"\btrivially (?:follows|holds|obtained)\b",
    r"\bstraightforward(?:ly)? (?:calculation|verification|check)\b",
    r"\bwe omit the (?:proof|derivation|details|calculation)\b",
    r"\bomit(?:ted)? (?:for brevity|here)\b",
    r"\bcan be shown (?:that|to)\b",
    r"\bone (?:can|may) (?:show|verify|check)\b",
    r"\ba (?:direct|simple) calculation (?:shows|gives|yields)\b",
    r"\bafter (?:some|a) (?:algebra|calculation|manipulation)\b",
    r"\bfollowing (?:the same|similar) (?:steps|arguments)\b",
    r"\bas shown in (?:the )?(?:appendix|section)\b",  # will be validated
    r"\bthe details are relegated\b",
    r"\bleft to the reader\b",
    r"\bclear(?:ly)? (?:that|from|follows)\b",
    r"\bobvious(?:ly)?\b",
]

BIO_TRIGGER_PATTERNS = [
    r"\bwe hypothesi[sz]e (?:that|the)\b",
    r"\bresults? (?:indicate|suggest|show|reveal)\b",
    r"\b(?:this|these) (?:results?|findings?) suggest (?:that|s)\b",
    r"\bconsistent with (?:the|a|our)\b",
    r"\bin (?:good |excellent )?agreement with\b",
    r"\bmay (?:play a role|contribute|reflect|indicate)\b",
    r"\bcorrelated? with\b",
    r"\bwe observed? (?:that|a|an)\b",
    r"\bappears? to\b",
    r"\blikely (?:reflects?|involves?|contributes?)\b",
    r"\bcannot (?:rule out|exclude)\b",
    r"\bfurther (?:studies|work|experiments) (?:are|will be) (?:needed|required)\b",
]

CHEM_TRIGGER_PATTERNS = [
    r"\bconfirmed by\b",
    r"\bas expected\b",
    r"\bsuccessfully (?:synthesi[sz]ed|demonstrated|prepared|obtained)\b",
    r"\bindicat(?:es|ing) the (?:presence|formation) of\b",
    r"\battributed to\b",
    r"\bthe enhanced (?:performance|activity|selectivity|stability)\b",
    r"\bin (?:excellent|good) agreement with\b",
    r"\bconsistent with (?:literature|previous|the)\b",
    r"\bsuggests? the (?:formation|presence) of\b",
    r"\bwe (?:propose|hypothesi[sz]e) (?:that|the)\b",
    r"\bstructure (?:was|is) confirmed\b",
    r"\bcan be (?:attributed|assigned) to\b",
]

MATH_TRIGGER_PATTERNS = [
    r"\bone (?:easily|readily) (?:sees|verifies|checks)\b",
    r"\ban argument (?:shows|gives)\b",
    r"\bsimilar to (?:the proof|the argument) of\b",
    r"\bwithout loss of generality\b",
    r"\bWLOG\b",
    r"\bit suffices to (?:show|verify|check|prove)\b",
    r"\bby (?:a standard|the same) argument\b",
    r"\bmutatis mutandis\b",
    r"\banalogous(?:ly)?\b",
    r"\bthe proof (?:is|will be) (?:similar|analogous)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in TRIGGER_PATTERNS]
_COMPILED_BIO = [re.compile(p, re.IGNORECASE) for p in BIO_TRIGGER_PATTERNS]
_COMPILED_CHEM = [re.compile(p, re.IGNORECASE) for p in CHEM_TRIGGER_PATTERNS]
_COMPILED_MATH = [re.compile(p, re.IGNORECASE) for p in MATH_TRIGGER_PATTERNS]


@dataclass
class Candidate:
    section_number: str
    section_title: str
    trigger_phrase: str
    context: str  # ~400 chars around the match
    offset: int   # position in section text


def scan_sections(parsed: dict, field: str = "physics") -> list[Candidate]:
    """Walk every section and pull trigger phrases with local context."""
    out: list[Candidate] = []
    if field == "biology":
        patterns = _COMPILED + _COMPILED_BIO
    elif field == "chemistry":
        patterns = _COMPILED + _COMPILED_CHEM
    elif field == "math":
        patterns = _COMPILED + _COMPILED_MATH
    else:
        patterns = _COMPILED

    for s in parsed.get("sections", []):
        text = s.get("text", "") or ""
        if len(text) < 100:
            continue
        seen_offsets: set[int] = set()
        for pat in patterns:
            for m in pat.finditer(text):
                start = max(0, m.start() - 200)
                end = min(len(text), m.end() + 200)
                # dedupe overlapping matches on the same line
                key = m.start() // 50
                if key in seen_offsets:
                    continue
                seen_offsets.add(key)
                out.append(Candidate(
                    section_number=s.get("number", ""),
                    section_title=s.get("title", ""),
                    trigger_phrase=m.group(0),
                    context=text[start:end].strip(),
                    offset=m.start(),
                ))
    return out


# ---------------------------------------------------------------------------
# Stage 2: LLM adjudication
# ---------------------------------------------------------------------------

STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "integer"},
                    "kind": {
                        "type": "string",
                        "enum": ["derivation_gap", "orphan_reference"],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "warning"],
                    },
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["candidate_id", "kind", "severity", "evidence", "confidence"],
            },
        },
    },
    "required": ["findings"],
}


STRUCTURE_SYSTEM = """You are a physics referee evaluating narrative and derivation
quality in a manuscript. You will be given a list of candidate 'trigger
phrases' extracted from the paper — phrases like 'it is easy to see',
'trivially follows', 'we omit the proof', 'as shown in the appendix'.

Your job: for each candidate, decide whether the phrase indicates a real
problem that a rigorous referee would flag, or whether it is an
acceptable rhetorical shortcut.

FINDING KINDS (only two — pick the closer fit):

- derivation_gap: A non-trivial step is claimed without justification.
  The referee would need to see the missing calculation or citation to
  accept the result. This ALSO covers hand-wavy shortcuts ('obvious',
  'clearly', 'trivial', 'easy') in contexts a competent reader in the
  target audience would NOT find obvious. Do not use a separate
  'handwavy' kind — hand-waving IS a derivation gap.

- orphan_reference: The paper says 'as shown in the appendix' or
  'see Section X' but the surrounding excerpt gives no concrete
  appendix or section number, or the referenced location is absent
  from the paper.

SEVERITY (only two):
- critical: A central claim of the paper rests on the omitted step.
- warning: The referee would ask the author to clarify before publication.

STRICT RULES:

1. Silence is the default. Physics writing routinely uses these phrases
   legitimately. If the surrounding context makes the step plausible for
   a physicist in the target audience, DO NOT flag it. An EMPTY findings
   list is the CORRECT answer in most cases.

2. Only flag a candidate when you can point to a SPECIFIC missing step:
   "the claim that <X> requires <Y> but the paper only shows <Z>".
   Vague concerns ("this could be more rigorous", "more detail would
   help") are NOT findings.

3. Evidence must name the specific missing step. Do NOT merely repeat
   the trigger phrase or the surrounding context.

4. Confidence must be >= 0.5. Anything below is speculation and should
   be omitted entirely.

5. Do NOT emit multiple findings for the same candidate_id. If a
   candidate is worth flagging, emit one finding for it.

Target: 0-4 findings per paper. Empty is common and correct. Never
exceed 6."""


def _format_candidates(cands: list[Candidate]) -> str:
    lines = []
    for i, c in enumerate(cands):
        lines.append(
            f"[{i}] section {c.section_number} — {c.section_title}\n"
            f"    trigger: \"{c.trigger_phrase}\"\n"
            f"    context: \"...{c.context}...\""
        )
    return "\n\n".join(lines)


def _format_paper_summary(parsed: dict) -> str:
    section_ids = ", ".join(
        f"{s.get('number','?')} {s.get('title','?')}"
        for s in parsed.get("sections", [])[:15]
    )
    return f"""TITLE:
{parsed.get('title', '')}

ABSTRACT:
{parsed.get('abstract', '')}

SECTIONS:
{section_ids}

EQUATION COUNT: {parsed.get('equation_count', 0)}"""


MAX_CANDIDATES_PER_CALL = 25


def analyze_structure(doi: str, client: Optional[GeminiClient] = None) -> StructureReport:
    if client is None:
        client = GeminiClient()

    report = StructureReport(doi_label=doi)
    parsed_path = get_path("processed_data") / "parsed" / f"{doi}.json"
    if not parsed_path.exists():
        return report
    parsed = json.loads(parsed_path.read_text())

    classification = _classification_for(doi)
    field_top = (classification or {}).get("field_top", "physics")
    candidates = scan_sections(parsed, field=field_top)
    report.n_candidates_scanned = len(candidates)

    if not candidates:
        return report

    # cap to avoid runaway prompts
    cands = candidates[:MAX_CANDIDATES_PER_CALL]

    user = f"""PAPER CONTEXT:

{_format_paper_summary(parsed)}


CANDIDATE TRIGGER PHRASES ({len(cands)}):

{_format_candidates(cands)}


Call record_structure_findings with your assessments. Include only the
candidates that are genuinely worth flagging; be conservative."""

    result = client.structured_call(
        model=CONFIG["gemini"]["model_specialist"],
        system=STRUCTURE_SYSTEM,
        user=user,
        tool_name="record_structure_findings",
        tool_description="Record structural / narrative referee findings for the paper.",
        tool_input_schema=STRUCTURE_SCHEMA,
        max_output_tokens=1500,
        fallback_models=CONFIG["gemini"].get("fallback_chain", []),
    )

    seen_cids: set[int] = set()
    for f in result.tool_input.get("findings", []):
        cid = f.get("candidate_id", -1)
        if not (0 <= cid < len(cands)):
            continue
        # Defensive: schema restricts kind + severity but strip drift.
        if f.get("severity") not in ("critical", "warning"):
            continue
        if f.get("kind") not in ("derivation_gap", "orphan_reference"):
            continue
        try:
            conf = float(f.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0.5:
            continue
        # Dedup: one finding per candidate_id.
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        c = cands[cid]
        report.findings.append(StructureFinding(
            kind=f["kind"],
            severity=f["severity"],
            location=f"Section {c.section_number} {c.section_title}".strip(),
            trigger_phrase=c.trigger_phrase,
            context_excerpt=c.context[:600],
            evidence=f["evidence"],
            suggested_fix=f.get("suggested_fix", ""),
            confidence=conf,
        ))

    report.n_findings = len(report.findings)
    report.n_critical = sum(1 for f in report.findings if f.severity == "critical")
    report.n_warning = sum(1 for f in report.findings if f.severity == "warning")
    report.model = result.model
    report.input_tokens = result.input_tokens
    report.output_tokens = result.output_tokens
    return report


def save_report(report: StructureReport, out_dir: Optional[Path] = None) -> Path:
    if out_dir is None:
        out_dir = get_path("processed_data") / "structure_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.doi_label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return out_path
