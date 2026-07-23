"""Smoke test: extract equations from one page of a SciPost PDF using Gemini Flash."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
load_dotenv()

import fitz  # PyMuPDF for rendering PDF pages to images
from google import genai
from google.genai import types
from rich.console import Console

console = Console()

# Configure Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Pick one paper and render an interior page (skip title page)
pdf_path = Path("data/raw/pdf/SciPostPhys.21.1.011.pdf")
doc = fitz.open(pdf_path)
console.print(f"[cyan]PDF:[/cyan] {pdf_path.name}  ({len(doc)} pages)")

# Render page 8 (0-indexed) — should have equations
page_num = 8
page = doc[page_num]
pix = page.get_pixmap(dpi=200)  # decent quality for math
image_bytes = pix.tobytes("png")
doc.close()

console.print(f"[cyan]Rendered page {page_num + 1} as PNG ({len(image_bytes)} bytes)[/cyan]")

# Ask Gemini to extract equations
prompt = """You are looking at a page from a physics paper. Extract every numbered
displayed equation you see on this page.

For each equation, return:
- equation_label: the parenthesized number, e.g. "(3)" or "(2.7)"
- latex: the equation body as valid LaTeX (do NOT include the label)
- context_hint: one short phrase describing what the equation represents,
  based on the surrounding text

Ignore inline math. Only capture DISPLAYED numbered equations.

Return strictly JSON in the shape:
{"equations": [
  {"equation_label": "(3)", "latex": "H = -J \\\\sum ...", "context_hint": "..."}
]}"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        prompt,
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0,
    ),
)

console.print(f"\n[bold]Raw response:[/bold]")
console.print(response.text[:3000])

# Try to parse
import json
try:
    data = json.loads(response.text)
    eqs = data.get("equations", [])
    console.print(f"\n[green]Extracted {len(eqs)} equations[/green]")
    for eq in eqs[:5]:
        console.print(f"\n  {eq.get('equation_label', '?')}: {eq.get('latex', '')[:120]}")
        console.print(f"    [dim]context: {eq.get('context_hint', '')}[/dim]")
except Exception as e:
    console.print(f"[red]JSON parse failed:[/red] {e}")

# Usage info
if hasattr(response, "usage_metadata"):
    u = response.usage_metadata
    console.print(f"\n[dim]Prompt tokens: {u.prompt_token_count}, "
                  f"output tokens: {u.candidates_token_count}, "
                  f"total: {u.total_token_count}[/dim]")
