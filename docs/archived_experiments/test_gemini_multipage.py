"""Test Gemini equation extraction across multiple pages to gauge coverage."""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
load_dotenv()

import fitz
from google import genai
from google.genai import types
from rich.console import Console
from rich.table import Table

console = Console()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

pdf_path = Path("data/raw/pdf/SciPostPhys.21.1.011.pdf")
doc = fitz.open(pdf_path)

# Sample 5 pages across the paper: intro area, main derivations, later math
sample_pages = [3, 9, 14, 20, 30]

EXTRACT_PROMPT = """You are looking at a page from a physics paper. Extract every numbered
displayed equation you see on this page.

For each equation, return:
- equation_label: the parenthesized number, e.g. "(3)" or "(2.7)"
- latex: the equation body as valid LaTeX (do NOT include the label)
- context_hint: one short phrase describing what the equation represents,
  based on the surrounding text

Ignore inline math. Only capture DISPLAYED numbered equations.
If the page has no displayed numbered equations, return an empty list.

Return strictly JSON in the shape:
{"equations": [
  {"equation_label": "(3)", "latex": "H = ...", "context_hint": "..."}
]}"""

table = Table(title=f"Coverage across pages of {pdf_path.name}")
table.add_column("Page", style="cyan", justify="right")
table.add_column("# eqs", style="green", justify="right")
table.add_column("Labels found", style="yellow")
table.add_column("Tokens (in/out)", style="dim")

total_in = 0
total_out = 0

for pnum in sample_pages:
    page = doc[pnum]
    pix = page.get_pixmap(dpi=200)
    image_bytes = pix.tobytes("png")

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            EXTRACT_PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    try:
        data = json.loads(resp.text)
        eqs = data.get("equations", [])
        labels = ", ".join(e.get("equation_label", "?") for e in eqs)
    except Exception:
        eqs = []
        labels = "PARSE_ERROR"

    u = resp.usage_metadata
    total_in += u.prompt_token_count
    total_out += u.candidates_token_count

    table.add_row(
        str(pnum + 1),  # 1-indexed
        str(len(eqs)),
        labels[:60],
        f"{u.prompt_token_count}/{u.candidates_token_count}",
    )

doc.close()
console.print(table)
console.print(f"\n[bold]Totals:[/bold] {total_in} in / {total_out} out tokens")

# Cost estimate for paid tier Gemini 2.5 Flash
# Prices: $0.30/M in, $2.50/M out (as of late 2025 for text; image tokens counted in input)
cost = (total_in / 1e6) * 0.30 + (total_out / 1e6) * 2.50
console.print(f"[dim]Cost for these 5 pages: ~${cost:.4f}[/dim]")
console.print(f"[dim]Extrapolated for full corpus (136 papers × avg 15 pages with equations): ~${cost * (136*15/5):.2f}[/dim]")
