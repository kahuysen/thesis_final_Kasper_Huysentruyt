"""OpenRouter-backed agent loop.

OpenRouter's API is OpenAI-compatible (same chat.completions surface +
function-calling schema), so we reuse the loop from `extractor_openai.py`.
This wrapper exists for symmetry with the other extractors and as the
construction point for the OpenRouter client.

Construction:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    for ev in extract_figure_stream_openrouter(image, model="anthropic/claude-opus-4-7",
                                               client=client, system_prompt=PROMPT):
        ...

Model ids on OpenRouter take the form `<provider>/<model>`, e.g.:
    anthropic/claude-opus-4-7
    openai/gpt-5.4
    google/gemini-3-pro-preview
    deepseek/deepseek-v4-flash
    openrouter/auto      (auto-select)

Vision caveat: only multimodal target models accept image input. The agent
sends the figure as an `image_url` content part — text-only models behind
OpenRouter will reject it the same way DeepSeek's text-only models would.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .extractor_openai import extract_figure_stream_openai


def extract_figure_stream_openrouter(
    image_path: str | Path,
    *,
    model: str,
    client,
    system_prompt: str,
    max_completion_tokens: int = 8192,
    max_steps: int = 128,
    extra_body: dict | None = None,
) -> Iterator[dict]:
    """Run the OpenRouter agent loop, yielding the same event dicts as the
    other extractors. Delegates to the OpenAI-compatible loop.

    Default `max_steps=128` because OpenRouter-fronted models (xAI Grok,
    Gemini-via-OR, etc.) emit one tool call per LLM round-trip *and* often
    interleave reasoning text — dense picture-style figures with 13+ entries
    blew through 64 in the first Grok run. 128 leaves comfortable headroom;
    the loop exits early on submit_extraction so this doesn't slow easy cases.
    """
    return extract_figure_stream_openai(
        image_path,
        model=model,
        client=client,
        system_prompt=system_prompt,
        max_completion_tokens=max_completion_tokens,
        max_steps=max_steps,
        extra_body=extra_body,
    )
