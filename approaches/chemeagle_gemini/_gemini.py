"""Gemini-via-OpenAI-compat-shim helper for ChemEagle-Gemini fork.

Centralizes the client+model so the fork can be retargeted with env vars
instead of touching code. Reads:
  GEMINI_API_KEY  (required) — also accepts API_KEY as fallback
  GEMINI_BASE_URL (default: Google's OpenAI-compat endpoint)
  GEMINI_MODEL    (default: gemini-3-flash-preview)

Also wraps chat.completions.create so that when both `tools=` and
`response_format=` are passed, response_format is dropped — Gemini's OpenAI
shim returns 400 ("Function calling with a response mime type: 'application/
json' is unsupported") if both are supplied.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any
from openai import OpenAI

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3-flash-preview"


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY (or API_KEY) must be set")
    return key


def get_base_url() -> str:
    return os.getenv("GEMINI_BASE_URL", DEFAULT_BASE_URL)


def get_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def _patch_create(client: OpenAI) -> OpenAI:
    """Wrap client.chat.completions.create to:
    1. Drop response_format when tools= is set (Gemini 400 otherwise).
    2. Disable Gemini's thinking on every call (reasoning_effort="none").
       Thinking models emit thought_signature on tool_calls, which the
       OpenAI-compat shim strips on output. When we echo the assistant
       message back in a multi-turn tool flow, Gemini rejects it with
       "Function call is missing a thought_signature in functionCall parts".
       Disabling thinking up-front avoids generating signatures we can't
       round-trip. Override per-call via reasoning_effort=... if needed.
    """
    completions = client.chat.completions
    orig = completions.create

    def wrapper(*args, **kwargs):
        if kwargs.get("tools") and kwargs.get("response_format"):
            print(
                f"[_gemini] dropping response_format={kwargs['response_format']} "
                f"because tools= is set (Gemini disallows the combo)",
                file=sys.stderr,
            )
            kwargs.pop("response_format", None)
        kwargs.setdefault("reasoning_effort", "none")
        return orig(*args, **kwargs)

    completions.create = wrapper  # type: ignore[assignment]
    return client


def get_client() -> OpenAI:
    return _patch_create(OpenAI(api_key=get_api_key(), base_url=get_base_url()))


MODEL = get_model()


def parse_json_content(content: str | None) -> Any:
    """Robust JSON parse for Gemini chat-completion content.

    Gemini frequently returns:
    - empty/None content when it produces tool calls instead of text
    - JSON wrapped in ```json ... ``` markdown fences
    - JSON followed by extra commentary ("Extra data" json error)
    - JSON arrays where the schema expected a single object (or vice-versa)
    Returns {} for empty/None content. Raises json.JSONDecodeError if no
    parseable JSON object or array can be found in the string.
    """
    if not content:
        return {}
    s = content.strip()
    if s.startswith("```"):
        m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", s, re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Bracket-balance scan: find the first balanced { ... } or [ ... ].
        for opener, closer in (("{", "}"), ("[", "]")):
            start = s.find(opener)
            if start < 0:
                continue
            depth = 0
            for i in range(start, len(s)):
                ch = s[i]
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(s[start:i + 1])
                        except json.JSONDecodeError:
                            break
        raise
