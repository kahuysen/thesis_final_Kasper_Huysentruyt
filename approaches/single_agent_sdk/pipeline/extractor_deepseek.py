"""DeepSeek-backed agent loop.

DeepSeek's API is OpenAI-compatible (same chat.completions surface + same
function-calling schema), so we reuse the loop from `extractor_openai.py`.
This thin wrapper exists for symmetry with the other extractors and as the
construction point for the DeepSeek client.

Construction:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )
    for ev in extract_figure_stream_deepseek(image, model="...", client=client,
                                             system_prompt=PROMPT):
        ...

Vision caveat: only multimodal DeepSeek models accept image input. The
text-only chat / reasoner models will reject the `image_url` content part
with a 400 / unsupported-content error. Confirm the model supports vision
before launching long runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .extractor_openai import extract_figure_stream_openai


def extract_figure_stream_deepseek(
    image_path: str | Path,
    *,
    model: str,
    client,
    system_prompt: str,
    max_completion_tokens: int = 8192,
) -> Iterator[dict]:
    """Run the DeepSeek agent loop, yielding the same event dicts as the
    other extractors. Delegates to the OpenAI-compatible loop."""
    return extract_figure_stream_openai(
        image_path,
        model=model,
        client=client,
        system_prompt=system_prompt,
        max_completion_tokens=max_completion_tokens,
    )
