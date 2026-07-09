"""Agent loop that turns a chemistry figure into a validated FigureExtraction.

The agent has two tools:
  - validate_smiles       (canonicalize, check parseable)
  - submit_extraction     (terminator — its argument IS the final JSON)

`submit_extraction`'s schema is the Pydantic FigureExtraction schema, so the
final tool call is guaranteed to be schema-valid.

Two entry points:
  - `extract_figure_stream(...)` — a generator that yields event dicts as the
    agent loop progresses. Used by the SSE endpoint and anyone who wants to
    surface live progress.
  - `extract_figure(...)` — the historical sync wrapper; consumes the stream
    internally and returns `(FigureExtraction, metadata)`.
"""
from __future__ import annotations

import base64
import copy
import json
import mimetypes
import re
from pathlib import Path
from typing import Iterator, Optional

from anthropic import Anthropic
from pydantic import ValidationError

from .config import (
    build_client,
    build_client_for,
    current_provider,
    default_model,
    provider_for_model,
)
from .schema import FigureExtraction
from .tools import TOOL_DISPATCH, TOOL_SCHEMAS

DEFAULT_MODEL = default_model()
MAX_AGENT_STEPS = 12  # safety cap on the tool-use loop


_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent / "prompts" / "SYSTEM_prompt_opus47.md"
)


def _load_system_prompt(path: Path) -> str:
    text = path.read_text()
    # Accept either raw prose or a `SYSTEM_PROMPT = """..."""` Python wrapper.
    m = re.search(r'"""(.+?)"""', text, re.DOTALL)
    return (m.group(1) if m else text).strip()


SYSTEM_PROMPT = _load_system_prompt(_PROMPT_FILE)


# ---------- helpers ----------

def _image_block(image_path: Path) -> dict:
    media_type, _ = mimetypes.guess_type(str(image_path))
    if media_type is None or not media_type.startswith("image/"):
        media_type = "image/png"
    data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _build_tool_schemas() -> list[dict]:
    """Patch the FigureExtraction Pydantic schema into submit_extraction."""
    schemas = copy.deepcopy(TOOL_SCHEMAS)
    fig_schema = FigureExtraction.model_json_schema()
    for s in schemas:
        if s["name"] == "submit_extraction":
            s["input_schema"] = fig_schema
    return schemas


def _format_args_preview(name: str, args: dict) -> str:
    """Short, human-readable rendering of the tool arguments for the UI panel."""
    args = args or {}
    if name == "submit_extraction":
        n = len(args.get("reactions") or [])
        return f"<{n} reactions>"
    if name == "validate_smiles":
        smi = args.get("smiles", "")
        return f'"{smi[:60]}"'
    s = json.dumps(args, separators=(",", ":"))
    return s if len(s) < 60 else s[:57] + "..."


def _format_result_summary(name: str, result: dict) -> str:
    """Telegraphic one-liner for the UI: 'ok · C₃H₇NO₂ · 89.0477'."""
    result = result or {}
    if not result.get("ok", False):
        err = result.get("error") or "error"
        return f"error · {str(err)[:80]}"
    if name == "validate_smiles":
        formula = result.get("molecular_formula", "?")
        mass = result.get("exact_mass", "?")
        return f"ok · {formula} · {mass}"
    if name == "submit_extraction":
        return "accepted"
    return "ok"


# ---------- streaming entry point ----------

def extract_figure_stream(
    image_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    provider: Optional[str] = None,
    client: Optional[Anthropic] = None,
) -> Iterator[dict]:
    """Run the agent loop, yielding events as it progresses.

    Event types (all dicts, JSON-serializable):
      * step_start  — {n, tool, args_preview, args}
      * step_done   — {n, tool, ok, summary, result}
      * complete    — {extraction, metadata}      (last event on success)
      * error       — {message}                   (last event on failure)

    `n` is the cumulative tool-call counter (1-indexed), independent of LLM turns.

    `provider` selects which backend handles this request (anthropic / azure /
    gemini). When omitted, it's inferred from the model id, falling back to
    LLM_PROVIDER. `client` overrides client construction (mainly for tests).
    """
    image_path = Path(image_path)

    # Resolve which provider services this request — driven by the model id
    # so the UI can mix providers in one dropdown.
    if provider is None:
        provider = provider_for_model(model) or current_provider()
    provider = provider.strip().lower()

    if client is None:
        client = build_client_for(provider)

    # Gemini speaks a different tool-calling protocol than Anthropic, so we
    # delegate the whole loop. Same event contract going out.
    if provider == "gemini":
        from .extractor_gemini_parallel import extract_figure_stream_gemini_parallel
        yield from extract_figure_stream_gemini_parallel(
            image_path,
            model=model,
            client=client,
            system_prompt=SYSTEM_PROMPT,
        )
        return

    if provider == "openrouter":
        from .extractor_openrouter import extract_figure_stream_openrouter
        yield from extract_figure_stream_openrouter(
            image_path,
            model=model,
            client=client,
            system_prompt=SYSTEM_PROMPT,
        )
        return

    tools = _build_tool_schemas()

    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                _image_block(image_path),
                {
                    "type": "text",
                    "text": (
                        f"Extract every reaction in this figure (filename: {image_path.name}). "
                        "Validate each SMILES, then call "
                        "`submit_extraction` exactly once with the final result."
                    ),
                },
            ],
        }
    ]

    submitted_dump: Optional[dict] = None
    tool_calls = 0
    input_tokens = 0
    output_tokens = 0
    last_step_index = 0

    try:
        for step in range(MAX_AGENT_STEPS):
            last_step_index = step
            resp = client.messages.create(
                model=model,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
            input_tokens += resp.usage.input_tokens
            output_tokens += resp.usage.output_tokens
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                break

            tool_results: list[dict] = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                tool_calls += 1
                name = block.name
                args = block.input or {}

                yield {
                    "event": "step_start",
                    "n": tool_calls,
                    "tool": name,
                    "args_preview": _format_args_preview(name, args),
                }

                if name == "submit_extraction":
                    try:
                        validated = FigureExtraction.model_validate(args)
                        submitted_dump = validated.model_dump()
                        result = {"ok": True, "accepted": True}
                    except ValidationError as e:
                        result = {
                            "ok": False,
                            "error": "schema validation failed",
                            "detail": e.errors()[:3],  # cap to keep wire small
                        }
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                            "is_error": not result.get("ok", False),
                        }
                    )
                elif name in TOOL_DISPATCH:
                    try:
                        result = TOOL_DISPATCH[name](**args)
                    except Exception as exc:
                        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                            "is_error": not result.get("ok", False),
                        }
                    )
                else:
                    result = {"ok": False, "error": f"unknown tool {name}"}
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                            "is_error": True,
                        }
                    )

                yield {
                    "event": "step_done",
                    "n": tool_calls,
                    "tool": name,
                    "ok": bool(result.get("ok")),
                    "summary": _format_result_summary(name, result),
                }

            messages.append({"role": "user", "content": tool_results})

            if submitted_dump is not None:
                # Submission accepted; no need for another LLM turn.
                break

        if submitted_dump is None:
            yield {
                "event": "error",
                "message": (
                    f"Agent did not call submit_extraction within "
                    f"{MAX_AGENT_STEPS} steps for {image_path.name}"
                ),
            }
            return

        metadata = {
            "image": str(image_path),
            "model": model,
            "steps": last_step_index + 1,
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        yield {"event": "complete", "extraction": submitted_dump, "metadata": metadata}

    except Exception as exc:
        # Don't crash the stream — surface as an error event so the UI can render it.
        yield {"event": "error", "message": f"{type(exc).__name__}: {exc}"}


# ---------- sync wrapper (back-compat) ----------

def extract_figure(
    image_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    provider: Optional[str] = None,
    client: Optional[Anthropic] = None,
    verbose: bool = False,
) -> tuple[FigureExtraction, dict]:
    """Run the agent on one figure. Returns (extraction, run_metadata).

    Thin wrapper around `extract_figure_stream`. Raises RuntimeError on
    failure (the streaming version surfaces the same condition as an
    `error` event)."""
    extraction_dump: Optional[dict] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None

    for ev in extract_figure_stream(image_path, model=model, provider=provider, client=client):
        kind = ev.get("event")
        if verbose and kind in ("step_start", "step_done"):
            arrow = "→" if kind == "step_start" else "←"
            print(f"  {arrow} step {ev['n']} {ev.get('tool','?')}  {ev.get('args_preview') or ev.get('summary','')}")
        if kind == "complete":
            extraction_dump = ev["extraction"]
            metadata = ev["metadata"]
        elif kind == "error":
            error = ev["message"]

    if error or extraction_dump is None:
        raise RuntimeError(error or "stream ended without a complete event")

    return FigureExtraction.model_validate(extraction_dump), metadata
