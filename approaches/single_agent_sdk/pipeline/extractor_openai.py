"""Azure-OpenAI-backed agent loop (GPT-5.4 etc.).

Mirrors the event contract of `extractor.extract_figure_stream`:
  step_start / step_done / complete / error
so the SSE endpoint and UI don't have to know which backend ran.

Why a separate file: OpenAI's chat-completions tool-call protocol differs
enough from Anthropic's tool-use blocks (`tool_calls` array vs. content
blocks, `role: "tool"` messages vs. user-role tool_results) that a clean
parallel implementation is easier to read than a multi-branch monolith.

Construction:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
    for ev in extract_figure_stream_openai(image, model=deployment_name,
                                           client=client, system_prompt=PROMPT):
        ...
"""
from __future__ import annotations

import base64
import copy
import json
import mimetypes
from pathlib import Path
from typing import Iterator, Optional

from pydantic import ValidationError

from .schema import FigureExtraction
from .tools import TOOL_DISPATCH, TOOL_SCHEMAS

MAX_AGENT_STEPS = 12


# ---------- helpers ----------

def _image_data_url(image_path: Path) -> str:
    # Sniff the real format from magic bytes — some benchmark images carry
    # the wrong extension (e.g. PNGs named .jpg), and strict providers
    # (Anthropic) reject a media type that contradicts the payload.
    raw = image_path.read_bytes()
    if raw.startswith(b"\x89PNG"):
        media_type = "image/png"
    elif raw[:2] == b"\xff\xd8":
        media_type = "image/jpeg"
    elif raw[:3] == b"GIF":
        media_type = "image/gif"
    elif raw[8:12] == b"WEBP":
        media_type = "image/webp"
    else:
        media_type, _ = mimetypes.guess_type(str(image_path))
        if media_type is None or not media_type.startswith("image/"):
            media_type = "image/png"
    data = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{media_type};base64,{data}"


def _build_openai_tools() -> list[dict]:
    """Translate our shared tool schemas into OpenAI's `tools` format."""
    schemas = copy.deepcopy(TOOL_SCHEMAS)
    fig_schema = FigureExtraction.model_json_schema()
    out: list[dict] = []
    for s in schemas:
        params = fig_schema if s["name"] == "submit_extraction" else s.get("input_schema", {})
        out.append(
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": params or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def _format_args_preview(name: str, args: dict) -> str:
    """Telegraphic argument preview shown in the UI tool-stream rail."""
    args = args or {}
    if name == "submit_extraction":
        n = len(args.get("reactions") or [])
        return f"<{n} reactions>"
    if name == "validate_smiles":
        return f'"{(args.get("smiles") or "")[:60]}"'
    s = json.dumps(args, separators=(",", ":"))
    return s if len(s) < 60 else s[:57] + "..."


def _format_result_summary(name: str, result: dict) -> str:
    result = result or {}
    if not result.get("ok", False):
        err = result.get("error") or "error"
        return f"error · {str(err)[:80]}"
    if name == "validate_smiles":
        return f"ok · {result.get('molecular_formula', '?')} · {result.get('exact_mass', '?')}"
    if name == "submit_extraction":
        return "accepted"
    return "ok"


# ---------- streaming entry point ----------

def extract_figure_stream_openai(
    image_path: str | Path,
    *,
    model: str,                  # Azure deployment name (e.g. "gpt-5.4")
    client,                      # openai.AzureOpenAI instance
    system_prompt: str,
    max_completion_tokens: int = 8192,
    max_steps: int = MAX_AGENT_STEPS,
    extra_body: dict | None = None,
) -> Iterator[dict]:
    """Run the OpenAI/Azure agent loop, yielding the same event dicts as the
    Anthropic version (step_start / step_done / complete / error).

    `extra_body` is passed through to `chat.completions.create` — used for
    OpenRouter provider routing (e.g. {"provider": {"order": [...]}})."""
    image_path = Path(image_path)
    tools = _build_openai_tools()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_path)},
                },
                {
                    "type": "text",
                    "text": (
                        f"Extract every reaction in this figure (filename: {image_path.name}). "
                        "Validate each SMILES, then call "
                        "`submit_extraction` exactly once with the final result."
                    ),
                },
            ],
        },
    ]

    submitted_dump: Optional[dict] = None
    tool_calls_n = 0
    input_tokens = 0
    output_tokens = 0
    last_step = 0
    providers: set[str] = set()   # upstream providers reported by OpenRouter

    try:
        for step in range(max_steps):
            last_step = step
            kwargs = dict(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            # GPT-5+ uses `max_completion_tokens` instead of `max_tokens`.
            kwargs["max_completion_tokens"] = max_completion_tokens
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as exc:
                # Newer SDKs may reject `max_completion_tokens` on legacy
                # deployments — fall back to the older arg.
                if "max_completion_tokens" in str(exc):
                    kwargs.pop("max_completion_tokens", None)
                    kwargs["max_tokens"] = max_completion_tokens
                    resp = client.chat.completions.create(**kwargs)
                else:
                    raise

            usage = getattr(resp, "usage", None)
            if usage is not None:
                input_tokens += usage.prompt_tokens or 0
                output_tokens += usage.completion_tokens or 0

            prov = getattr(resp, "provider", None)
            if isinstance(prov, str) and prov:
                providers.add(prov)

            choice = resp.choices[0]
            msg = choice.message
            # Append assistant message verbatim — must match what we send back.
            assistant_entry: dict = {
                "role": "assistant",
                "content": msg.content,
            }
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_entry)

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                tool_calls_n += 1
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                yield {
                    "event": "step_start",
                    "n": tool_calls_n,
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
                            "detail": e.errors()[:3],
                        }
                elif name in TOOL_DISPATCH:
                    try:
                        result = TOOL_DISPATCH[name](**args)
                    except Exception as exc:
                        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                else:
                    result = {"ok": False, "error": f"unknown tool {name}"}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

                yield {
                    "event": "step_done",
                    "n": tool_calls_n,
                    "tool": name,
                    "ok": bool(result.get("ok")),
                    "summary": _format_result_summary(name, result),
                }

            if submitted_dump is not None:
                break

        if submitted_dump is None:
            yield {
                "event": "error",
                "message": (
                    f"Agent did not call submit_extraction within "
                    f"{max_steps} steps for {image_path.name}"
                ),
            }
            return

        metadata = {
            "image": str(image_path),
            "model": model,
            "steps": last_step + 1,
            "tool_calls": tool_calls_n,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "providers": sorted(providers),
        }
        yield {"event": "complete", "extraction": submitted_dump, "metadata": metadata}

    except Exception as exc:
        yield {"event": "error", "message": f"{type(exc).__name__}: {exc}"}
