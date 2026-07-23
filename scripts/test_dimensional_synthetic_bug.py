"""Adversarial test: inject known dimensional bugs and verify detection.

We take a real paper's claim/symbol data, corrupt it in specific ways, then
run the dimensional checker on the corrupted version. A good checker should
raise severity=critical or =warning verdicts on the injected bugs.
"""
import sys
import json
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.dimensional_checker import check_paper
from peerlens.settings import get_path
from rich.console import Console
from rich.table import Table

console = Console()

doi = "SciPostPhys.21.1.011"
claims_path = get_path("processed_data") / "claims" / f"{doi}.json"
pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
original = json.loads(claims_path.read_text())


def make_bug_energy_vs_length():
    """BUG 1: H_M declared 'length' instead of 'energy' — sum with h_m makes no sense.
    Detection expected: critical/inconsistent — length + operator + dimensionless β can't form energy.
    """
    d = copy.deepcopy(original)
    for s in d["symbols"]:
        if s["symbol"] == "H_M":
            s["units"] = "length (m)"  # was: energy
            s["meaning"] += "  [CORRUPTED: was 'energy']"
    d["_injected_bug"] = "H_M declared as length instead of energy"
    return d


def make_bug_argument_of_exp():
    """BUG 2: add a synthetic claim that violates 'exp argument must be dimensionless'.
    Detection expected: critical/inconsistent.
    """
    d = copy.deepcopy(original)
    d["claims"].append({
        "claim_type": "dimensional",
        "location": "Eq. (synthetic)",
        "referenced_equations": ["synth1"],
        "natural_language": "The partition function contains exp(H_M) where H_M has units of energy.",
        "check_hint": "Argument of exp must be dimensionless — but H_M has energy units.",
        "verification_method": "sympy",
    })
    d["_injected_bug"] = "exp(H_M) with dimensional argument"
    return d


def make_bug_symbol_conflict():
    """BUG 3: define b_m twice with different units."""
    d = copy.deepcopy(original)
    d["symbols"].append({
        "symbol": "b_m",
        "meaning": "COMPETING definition of b_m as energy scale",
        "units": "energy (eV)",
        "first_seen_at": "Eq. (synthetic)",
    })
    d["_injected_bug"] = "b_m redefined with conflicting units"
    return d


def make_bug_sum_mismatch():
    """BUG 4: claim explicitly says two terms in a sum have different units."""
    d = copy.deepcopy(original)
    d["claims"].append({
        "claim_type": "dimensional",
        "location": "Eq. (synthetic-2)",
        "referenced_equations": ["synth2"],
        "natural_language": "H_total = H_M + T where T is the temperature in Kelvin.",
        "check_hint": "H_M is an operator with energy units; T is a scalar in K. These cannot be added.",
        "verification_method": "sympy",
    })
    d["_injected_bug"] = "sum of Hamiltonian operator and temperature scalar"
    return d


bugs = {
    "BUG_1_H_M_length": make_bug_energy_vs_length,
    "BUG_2_exp_arg": make_bug_argument_of_exp,
    "BUG_3_b_m_conflict": make_bug_symbol_conflict,
    "BUG_4_sum_mismatch": make_bug_sum_mismatch,
}

results = Table(title="Adversarial detection results", show_lines=True)
results.add_column("Bug", style="cyan")
results.add_column("Detected?", style="bold")
results.add_column("Severity")
results.add_column("Verdict")
results.add_column("Evidence excerpt", style="dim", max_width=60)

for name, factory in bugs.items():
    console.print(f"\n[yellow]Running[/yellow] {name}")
    corrupted = factory()
    report = check_paper(corrupted, pdf_path, max_vision_calls=0)

    # Look for any finding that names the injected concern
    detected_severity = "-"
    detected_verdict = "-"
    detected_evidence = "-"
    hit = False
    keywords = {
        "BUG_1_H_M_length": ["length", "H_M"],
        "BUG_2_exp_arg": ["exp", "dimensionless"],
        "BUG_3_b_m_conflict": ["b_m", "conflict", "redefined", "different"],
        "BUG_4_sum_mismatch": ["temperature", "H_M + T", "sum"],
    }[name]
    for f in report.findings:
        ev_low = f.evidence.lower()
        if any(k.lower() in ev_low for k in keywords) and f.verdict in ("inconsistent", "convention_dependent"):
            hit = True
            detected_severity = f.severity
            detected_verdict = f.verdict
            detected_evidence = f.evidence[:200]
            break
    if not hit:
        # fallback: check for any critical/warning finding at all
        for f in report.findings:
            if f.severity in ("critical", "warning"):
                detected_severity = f.severity + " (weak match)"
                detected_verdict = f.verdict
                detected_evidence = f.evidence[:200]
                hit = "partial"
                break

    results.add_row(
        name,
        "YES" if hit is True else ("PARTIAL" if hit == "partial" else "NO"),
        detected_severity,
        detected_verdict,
        detected_evidence,
    )

console.print(results)
