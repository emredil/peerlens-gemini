"""Smoke test: send a physics PDF page to Gemini and extract equations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google import genai
from google.genai import types

from peerlens.settings import google_api_key, get_path
from rich.console import Console

console = Console()

# Take one PDF from our corpus
pdf_path = get_path("raw_data") / "pdf" / "SciPostPhys.21.1.011.pdf"
console.print(f"[cyan]Testing Gemini vision on:[/cyan] {pdf_path.name}")

client = genai.Client(api_key=google_api_key())

pdf_bytes = pdf_path.read_bytes()
console.print(f"PDF size: {len(pdf_bytes) / 1024:.1f} KB")

prompt = """You are a physics equation extractor.
Look at this PDF. Find the first 5 numbered equations.
For each, return:
- Equation number as shown in the paper
- The equation transcribed as LaTeX
- The page it appears on
- One-sentence description of what the equation says

Respond in JSON like:
{
  "equations": [
    {"number": "1", "latex": "...", "page": 3, "description": "..."}
  ]
}
Return only JSON, no other text."""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        prompt,
    ],
)

console.rule("[bold]Response[/bold]")
console.print(response.text)

console.rule("[bold]Usage[/bold]")
console.print(f"Input tokens:  {response.usage_metadata.prompt_token_count}")
console.print(f"Output tokens: {response.usage_metadata.candidates_token_count}")
console.print(f"Total tokens:  {response.usage_metadata.total_token_count}")
