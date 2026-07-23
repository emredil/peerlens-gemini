"""Adversarial test for SymPy verifier: inject bugs into equation LaTeX and
verify the checker catches them as 'refuted'.

Strategy: mutate the equation LaTeX used by the verifier so that the claim
no longer matches the (corrupted) equations. A trustworthy verifier should
flip its verdict from 'verified' to 'refuted'.
"""
import sys
import json
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.settings import get_path
from peerlens.agents.sympy_verifier import build_snippet, run_snippet
from peerlens.agents.equation_extractor import extract_equations_for_claim
from anthropic import Anthropic
from rich.console import Console
from rich.table import Table

console = Console()
client = Anthropic()

doi = "SciPostPhys.21.1.011"
pdf = get_path("raw_data") / "pdf" / f"{doi}.pdf"
claims = json.loads((get_path("processed_data") / "claims" / f"{doi}.json").read_text())
sympy_claims = [(i, c) for i, c in enumerate(claims["claims"])
                if c.get("verification_method") == "sympy"]

# --- Test cases ---
# Each test targets a specific claim + a corrupted LaTeX version of it.
# claim_id is the position in claims['claims'] (0-indexed by extractor order).

test_cases = []


def find_claim(loc_prefix: str, claim_type: str = None):
    for i, c in sympy_claims:
        if c.get("location", "").startswith(loc_prefix):
            if claim_type is None or c["claim_type"] == claim_type:
                return i, c
    return None, None


# BUG A: break the reduction — swap 0 with 1 in β. Setting β=1 (not 0) should
# NOT recover H_FFD, so the claim's substitution is now false.
i, c = find_claim("Equation (6), Section 2.1", "reduction")
if c is not None:
    corrupted_claim = copy.deepcopy(c)
    corrupted_claim["natural_language"] = corrupted_claim["natural_language"].replace(
        "β_{2m-1} = 0", "β_{2m-1} = 1"
    ).replace("vanish", "become one")
    test_cases.append(("BUG_A_wrong_substitution", corrupted_claim, {
        "6": "H_{FFD} = \\sum_{m=1}^{M} h_m",
        "7": "H_M = \\sum_{m=1}^{M} h_m + \\sum_{m=2}^{\\lfloor M/2 \\rfloor} \\beta_{2m-3} h_{2m-2} h_{2m-3} h_{2m}",
    }))

# BUG B: corrupt Eq (7) LaTeX itself so it no longer reduces to (6) when β=0.
# We change one of the h_m in the first sum to h_m^2 — makes H_FFD comparison fail.
i, c = find_claim("Equation (6), Section 2.1", "reduction")
if c is not None:
    corrupted_latex_7 = "H_M = \\sum_{m=1}^{M} h_m^2 + \\sum_{m=2}^{\\lfloor M/2 \\rfloor} \\beta_{2m-3} h_{2m-2} h_{2m-3} h_{2m}"
    test_cases.append(("BUG_B_corrupted_eq7", c, {
        "6": "H_{FFD} = \\sum_{m=1}^{M} h_m",
        "7": corrupted_latex_7,
    }))

# BUG C: break the recursion relation Eq (8) — swap the sign in the numerator.
# The claim quotes the formula; if the LaTeX now shows a different formula,
# it should either refute or emit unable_to_verify, but NOT verified.
i, c = find_claim("Equation (8), Section 2.2")
if c is not None:
    bogus_eq8 = "\\beta_{2m+1} = \\frac{\\beta_{2m-1} + 100}{b_{2m-2}^{2} - \\beta_{2m-3} + 1}"
    test_cases.append(("BUG_C_wrong_recursion_formula", c, {"8": bogus_eq8}))

# --- Run each ---
symbols_catalogue = claims.get("symbols", [])

results = Table(title="SymPy adversarial detection", show_lines=True)
results.add_column("Bug", style="cyan")
results.add_column("Verdict", style="bold")
results.add_column("Correct?")
results.add_column("Detail", style="dim", max_width=60)

for name, claim, eq_latex in test_cases:
    console.print(f"\n[yellow]Running[/yellow] {name}")
    snippet, reasoning, conf, in_t, out_t = build_snippet(
        claim, symbols_catalogue, eq_latex, client,
    )
    verdict, detail, _, _ = run_snippet(snippet)

    # A trustworthy checker should NOT say 'verified' here
    good = verdict != "verified"
    results.add_row(
        name,
        verdict,
        "PASS" if good else "FAIL",
        f"reasoning: {reasoning[:120]} | verdict detail: {detail[:120]}",
    )

console.print(results)
