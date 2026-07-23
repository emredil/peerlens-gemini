"""Structure & Narrative Agent (Gemini port).

Detects derivation gaps, hand-wavy proof shortcuts, and undefined-argument
patterns that human referees typically flag as 'please clarify' or 'please
provide the missing derivation'.

Two stages:
  1. Local pattern scan — locate candidate 'trigger' phrases in section
     bodies (e.g. "trivially follows", "it is easy to see", "we omit",
     "as shown in", "straightforward calculation shows", ...).
  2. LLM adjudication — for each candidate, ask Gemini pro-latest to judge
     whether the phrase indicates a real derivation gap in context, or is
     an acceptable rhetorical shortcut. This filters out benign uses.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from peerlens.settings import CONFIG, get_path
from peerlens.gemini_client import GeminiClient


@dataclass
class StructureFinding:
    kind: str          # "derivation_gap" | "handwavy_shortcut" | "orphan_reference" | "info"
    severity: str      # "critical" | "warning" | "info"
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
    n_info: int = 0
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

_COMPILED = [re.compile(p, re.IGNORECASE) for p in TRIGGER_PATTERNS]


@dataclass
class Candidate:
    section_number: str
    section_title: str
    trigger_phrase: str
    context: str  # ~400 chars around the match
    offset: int   # position in section text


def scan_sections(parsed: dict) -> list[Candidate]:
    """Walk every section and pull trigger phrases with local context."""
    out: list[Candidate] = []
    for s in parsed.get("sections", []):
        text = s.get("text", "") or ""
        if len(text) < 100:
            continue
        seen_offsets: set[int] = set()
        for pat in _COMPILED:
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
                        "enum": ["derivation_gap", "handwavy_shortcut", "orphan_reference", "info"],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "warning", "info"],
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
'trivially follows', 'we omit the proof', 'as shown in the appendix', etc.

Your job: for each candidate, decide whether the phrase indicates a real
derivation gap that a rigorous referee would flag, or whether it is an
acceptable rhetorical shortcut given the paper's level and context.

FINDING KINDS:

- derivation_gap: A non-trivial step is claimed without argument. The
  referee would need to see the missing calculation or citation to
  accept the result. Severity typically 'warning'; use 'critical' only
  if a central claim rests on the omitted step.

- handwavy_shortcut: The paper uses 'obvious', 'clearly', 'easy',
  'trivial' etc. in a context that is genuinely NOT obvious to a
  competent reader in the field. Severity 'warning' or 'info'.

- orphan_reference: The paper says 'as shown in the appendix' or 'see
  Section X' but the surrounding excerpt does not name a concrete
  appendix/section, or the referenced location seems inappropriate.
  Severity 'info' unless the reference is central.

- info: The phrase appears but its use is acceptable — the paper does
  provide enough surrounding context, or the step really is trivial for
  a physicist in the target audience. Include a brief note explaining
  why this is fine.

STRICTNESS:

Be conservative. Physics writing routinely uses these phrases in fully
legitimate ways. Only flag as 'derivation_gap' or 'handwavy_shortcut' if
you can point to a specific concrete step that a competent referee would
demand. If in doubt, use severity 'info' with kind 'info'.

Aim for at most 5-10 findings across the whole manuscript. Return an
empty list if the candidates you receive are all acceptable uses."""


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

    candidates = scan_sections(parsed)
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
        max_output_tokens=3500,
        fallback_models=CONFIG["gemini"].get("fallback_chain", []),
    )

    for f in result.tool_input.get("findings", []):
        cid = f.get("candidate_id", -1)
        if not (0 <= cid < len(cands)):
            continue
        c = cands[cid]
        report.findings.append(StructureFinding(
            kind=f["kind"],
            severity=f["severity"],
            location=f"Section {c.section_number} {c.section_title}".strip(),
            trigger_phrase=c.trigger_phrase,
            context_excerpt=c.context[:600],
            evidence=f["evidence"],
            suggested_fix=f.get("suggested_fix", ""),
            confidence=f.get("confidence", 0.0),
        ))

    report.n_findings = len(report.findings)
    report.n_critical = sum(1 for f in report.findings if f.severity == "critical")
    report.n_warning = sum(1 for f in report.findings if f.severity == "warning")
    report.n_info = sum(1 for f in report.findings if f.severity == "info")
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
