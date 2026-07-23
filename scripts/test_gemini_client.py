import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.gemini_client import GeminiClient
from rich.console import Console

console = Console()

client = GeminiClient()

schema = {
    "type": "object",
    "properties": {
        "subdomain": {"type": "string", "description": "physics subdomain code"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["subdomain", "rationale"],
}

result = client.structured_call(
    model="gemini-3-flash-preview",
    system="You classify physics abstracts by subdomain (condensed_matter, high_energy, quantum, gravitation, etc).",
    user="Classify this abstract:\n\nWe study a novel model of interacting spin chains with hidden free-fermionic structure using algebraic methods, deriving exactly solvable Hamiltonians via non-standard Jordan-Wigner transformations.",
    tool_name="classify",
    tool_description="Return the subdomain classification for the abstract.",
    tool_input_schema=schema,
    max_output_tokens=1024,
)

console.rule("[bold]Gemini structured call test[/bold]")
console.print(f"Model:          {result.model}")
console.print(f"Tokens in/out:  {result.input_tokens}/{result.output_tokens}")
console.print(f"Fallback used:  {result.fallback_used}")
console.print(f"Tool input:")
console.print(result.tool_input)
