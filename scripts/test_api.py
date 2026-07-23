"""Smoke test: verify Anthropic API connectivity and model availability."""
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()  # picks up ANTHROPIC_API_KEY from env

print("Testing Anthropic API connection...\n")

response = client.messages.create(
    model="claude-haiku-4-5-20251001",  # cheapest model for smoke test
    max_tokens=200,
    messages=[
        {
            "role": "user",
            "content": (
                "You are a physics reviewer smoke test. "
                "Respond with exactly one sentence confirming you can classify "
                "a physics paper by subdomain."
            ),
        }
    ],
)

print("Response:")
print(response.content[0].text)
print("\n--- Usage ---")
print(f"Input tokens:  {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Model:         {response.model}")
print("\nAPI connection: OK")
