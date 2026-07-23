"""Gemini adapter — provides a Claude-like structured-output API.

Rationale: our agents were written against Anthropic's tool-use API. Rather
than rewrite every agent for Gemini's function-calling syntax, we expose a
small compatibility layer here so the agents can keep their shape.

Public surface:
    client = GeminiClient()
    result = client.structured_call(
        model="gemini-3-flash-preview",
        system="You are ...",
        user="Extract claims from ...",
        tool_name="record_math_claims",
        tool_description="Record all verifiable claims.",
        tool_input_schema={...},           # JSON Schema, same shape as Anthropic
        max_output_tokens=4096,
        images=[{"mime": "image/png", "data": <bytes>}],  # optional
    )
    # result: {"tool_input": {...}, "input_tokens": int, "output_tokens": int, "model": str}
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_UNSUPPORTED_SCHEMA_KEYS = {"additionalProperties", "additional_properties", "$schema", "$id", "$ref"}


def _sanitize_schema_for_gemini(schema: Any) -> Any:
    """Strip JSON Schema keywords Gemini's OpenAPI subset does not accept."""
    if isinstance(schema, dict):
        return {
            k: _sanitize_schema_for_gemini(v)
            for k, v in schema.items()
            if k not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_sanitize_schema_for_gemini(x) for x in schema]
    return schema

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass
class StructuredResult:
    tool_input: dict
    input_tokens: int
    output_tokens: int
    model: str
    raw_text: str = ""
    fallback_used: bool = False




def _attempt_partial_json(text: str) -> dict:
    """Try to salvage a truncated JSON object by trimming to the last complete
    structural boundary and closing dangling brackets.

    Not perfect — designed only for our known top-level shape
    {"findings": [ ... ]} produced by function-call fallback.
    """
    if not text.strip():
        return {}

    s = text.strip()

    # Trim to the last complete object inside the "findings" array, if we
    # can identify the pattern.
    findings_start = s.find('"findings"')
    if findings_start == -1:
        return {}

    array_start = s.find("[", findings_start)
    if array_start == -1:
        return {}

    # Walk the string tracking brace/bracket depth and string state to find
    # the position of the last complete finding object.
    depth = 0
    in_str = False
    escape = False
    last_complete = -1

    i = array_start
    while i < len(s):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 1:
                    # Just closed a top-level finding object
                    last_complete = i
            elif ch == "]" and depth == 1:
                # This is the outer findings-array closing bracket
                # already inside a complete structure — should be safe
                last_complete = i
                break
        i += 1

    if last_complete == -1:
        return {"findings": []}

    salvaged = s[: last_complete + 1] + "]}"
    try:
        return json.loads(salvaged)
    except json.JSONDecodeError:
        return {"findings": []}


class GeminiClient:
    """Thin wrapper that exposes a Claude-like structured-call API.

    Two paths:
      1. Function calling — Gemini's native structured output. We declare a
         single function whose parameters mirror the requested JSON schema
         and force the model to call it.
      2. JSON-schema response mode — used as a fallback if function calling
         yields no call. The model returns a JSON object matching the schema.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing in .env")
        self._client = genai.Client(api_key=key)

    # ------------------------------------------------------------------
    # Structured call
    # ------------------------------------------------------------------
    def structured_call(
        self,
        model: str,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        tool_input_schema: dict,
        max_output_tokens: int = 4096,
        images: Optional[list[dict]] = None,
        temperature: float = 0.2,
        retries: int = 3,
        retry_sleep: float = 2.0,
        fallback_models: Optional[list[str]] = None,
    ) -> StructuredResult:
        """Call a Gemini model with a forced structured JSON output.

        Args:
            tool_input_schema: JSON Schema (dict) describing the desired
                structured output. Same format we've been using for
                Anthropic tools.
        """
        # Build parts
        parts: list[Any] = []
        if images:
            for img in images:
                parts.append(types.Part.from_bytes(mime_type=img["mime"], data=img["data"]))
        parts.append(types.Part(text=user))
        contents = [types.Content(role="user", parts=parts)]

        # Primary path: function calling
        sanitized_schema = _sanitize_schema_for_gemini(tool_input_schema)
        function_decl = types.FunctionDeclaration(
            name=tool_name,
            description=tool_description,
            parameters=sanitized_schema,
        )
        tools = [types.Tool(function_declarations=[function_decl])]
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=[tool_name],
            )
        )
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=tools,
            tool_config=tool_config,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        candidate_models = [model] + list(fallback_models or [])
        last_err = None
        for candidate in candidate_models:
            for attempt in range(retries):
                try:
                    resp = self._client.models.generate_content(
                        model=candidate,
                        contents=contents,
                        config=config,
                    )
                    tool_input = self._extract_function_call(resp)
                    usage = getattr(resp, "usage_metadata", None)
                    in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
                    out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

                    if tool_input is not None:
                        return StructuredResult(
                            tool_input=tool_input,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            model=candidate,
                            raw_text="",
                        )

                    fallback = self._json_schema_fallback(
                        candidate, system, user, images, tool_input_schema,
                        max_output_tokens, temperature,
                    )
                    return fallback

                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    if any(t in msg for t in ("429", "503", "quota", "unavailable", "resource exhausted")):
                        time.sleep(retry_sleep * (attempt + 1))
                        continue
                    if "404" in msg or "not found" in msg or "no longer available" in msg:
                        break
                    break
            # Try next candidate model

        raise RuntimeError(f"Gemini structured_call failed on all candidates ({candidate_models}): {last_err}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_function_call(self, resp) -> Optional[dict]:
        """Pull the first function_call from the response, if any."""
        for candidate in getattr(resp, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "args", None) is not None:
                    args = fc.args
                    if hasattr(args, "items"):
                        return dict(args)
                    try:
                        return json.loads(args) if isinstance(args, str) else dict(args)
                    except Exception:
                        continue
        return None

    def _json_schema_fallback(
        self,
        model: str,
        system: str,
        user: str,
        images: Optional[list[dict]],
        schema: dict,
        max_output_tokens: int,
        temperature: float,
    ) -> StructuredResult:
        """Ask the model to emit JSON directly, matching the schema."""
        parts: list[Any] = []
        if images:
            for img in images:
                parts.append(types.Part.from_bytes(mime_type=img["mime"], data=img["data"]))
        parts.append(types.Part(text=user))
        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            system_instruction=system + "\n\nReturn ONLY a JSON object matching this schema. No prose, no code fences.",
            response_mime_type="application/json",
            response_schema=_sanitize_schema_for_gemini(schema),
            temperature=temperature,
            max_output_tokens=max(max_output_tokens, 8192),
        )
        resp = self._client.models.generate_content(model=model, contents=contents, config=config)
        text = ""
        try:
            text = resp.text or ""
        except Exception:
            for c in getattr(resp, "candidates", []) or []:
                for p in getattr(c.content, "parts", []) or []:
                    if hasattr(p, "text"):
                        text += p.text or ""
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
            if stripped.endswith("```"):
                stripped = stripped[: stripped.rfind("```")]
            stripped = stripped.strip()

        data: dict = {}
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            # Attempt to recover truncated JSON by closing dangling structures
            data = _attempt_partial_json(stripped)

        usage = getattr(resp, "usage_metadata", None)
        return StructuredResult(
            tool_input=data,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            model=model,
            raw_text=text,
            fallback_used=True,
        )

    def plain_text(
        self,
        model: str,
        system: str,
        user: str,
        max_output_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens) for simple prompts."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        resp = self._client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part(text=user)])],
            config=config,
        )
        text = ""
        try:
            text = resp.text or ""
        except Exception:
            for c in getattr(resp, "candidates", []) or []:
                for p in getattr(c.content, "parts", []) or []:
                    if hasattr(p, "text"):
                        text += p.text or ""
        usage = getattr(resp, "usage_metadata", None)
        return (
            text,
            getattr(usage, "prompt_token_count", 0) if usage else 0,
            getattr(usage, "candidates_token_count", 0) if usage else 0,
        )
