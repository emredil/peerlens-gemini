"""Smoke test: extract equations from one PDF page using Gemini (new google-genai SDK)."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
console = Console()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    console.print("[red]GEMINI_API_KEY missing[/red]")
    sys.exit(1)

client = genai.Client(api_key=api_key)

CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-pro",
]

from peerlens.settings import get_path

doi = "SciPostPhys.21.1.011"
pdf_path = get_path("raw_data") / "pdf" / f"{doi}.pdf"
TARGET_PAGE = 4

doc = fitz.open(pdf_path)
console.print(f"[cyan]PDF pages:[/cyan] {len(doc)}")
if TARGET_PAGE >= len(doc):
    TARGET_PAGE = 3
page = doc[TARGET_PAGE]
pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
png_bytes = pix.tobytes("png")
doc.close()

snap_path = Path("/tmp") / f"{doi}_page{TARGET_PAGE+1}.png"
snap_path.write_bytes(png_bytes)
console.print(f"[dim]Snapshot: {snap_path}[/dim]")

PROMPT = """You are an OCR assistant for physics manuscripts.

Extract EVERY numbered equation from this page as a JSON list.
Each item: {"number": "7", "latex": "..."}

Rules:
- Preserve exact LaTeX with \\sum, \\beta, subscripts {}, superscripts ^{}.
- Overlines as \\bar{...}.
- ONLY numbered equations (labelled (N) or (X.Y)).
- Return ONLY the JSON list, no code fences, no prose.
- If none visible, return []."""

image_part = types.Part.from_bytes(data=png_bytes, mime_type="image/png")

for model_name in CANDIDATE_MODELS:
    console.rule(f"[bold]Trying {model_name}[/bold]")
    try:
        t0 = time.time()
        response = client.models.generate_content(
            model=model_name,
            contents=[PROMPT, image_part],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )
        dt = time.time() - t0

        text = response.text or ""
        console.print(f"[green]Success[/green] in {dt:.2f}s")
        usage = getattr(response, "usage_metadata", None)
        if usage:
            console.print(f"[dim]Tokens: prompt={getattr(usage,'prompt_token_count','?')}, "
                          f"out={getattr(usage,'candidates_token_count','?')}[/dim]")

        console.print("\n[bold]Response:[/bold]")
        console.print(text[:3000])
        if len(text) > 3000:
            console.print("[dim]... truncated[/dim]")
        break

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        console.print(f"[red]Failed[/red]: {err[:400]}")
        s = err.lower()
        if any(t in s for t in ["quota", "429", "resource exhausted",
                                 "unavailable", "503", "500", "rate", "not found", "404"]):
            console.print("[yellow]→ next model[/yellow]")
            time.sleep(1)
            continue
        console.print("[red]Non-retryable, aborting.[/red]")
        break
else:
    console.print("\n[red]All Gemini models failed.[/red]")
