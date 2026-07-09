"""Gemini-backed agent loop with parallel function calling.

Same event contract as the other extractors:
  step_start / step_done / complete / error

Why a separate file from the Anthropic / OpenAI extractors: Gemini's
function-calling protocol differs from Anthropic's tool-use blocks
(different schema field names, different message-history shape, different
terminator), so a clean parallel implementation is easier to read than a
multi-branch monolith.

Parallel function calling notes:
  - Gemini 3 Flash Preview naturally emits multiple `validate_smiles`
    calls per response when the calls are independent. The agent loop
    below collects every function-call part from each response and
    replies with one matching `function_response` part per call in the
    same user turn. A 14-entry table typically completes in 2-3
    round-trips and a 13-entry dense picture in ~2.
  - `automatic_function_calling=disable=True` keeps the manual loop
    authoritative — without it the SDK may silently serialise parallel
    calls into separate generate_content invocations on our behalf.
  - The function-response Part echoes the call's `id` back via
    `types.FunctionResponse(id=fc.id, ...)`. Per the Gemini 3 docs:
    "Always include the exact id from the function_call in your
    function_response so the API can map the result to the correct
    request." The SDK falls back to positional matching when id is None,
    but being explicit is safer for parallel batches.
    (`Part.from_function_response(...)` doesn't accept `id` in
    google-genai 1.74.0, hence the explicit construction.)
"""
from __future__ import annotations

import copy
import json
import mimetypes
from pathlib import Path
from typing import Iterator, Optional

from pydantic import ValidationError

from .schema import FigureExtraction
from .tools import TOOL_DISPATCH, TOOL_SCHEMAS

# Round-trip cap. With parallel calling, 30+ round-trips are unnecessary;
# 32 leaves headroom for the worst image. The loop exits early on
# submit_extraction so this doesn't slow easy cases.
MAX_AGENT_STEPS = 32


# ---------- schema translation ----------

# Gemini's function declarations don't accept JSON-Schema's `default` /
# `additionalProperties` / `$defs` etc. We keep the structural keys it does
# understand and drop the rest recursively.
_GEMINI_SCHEMA_KEYS = {
    "type",
    "format",
    "description",
    "enum",
    "items",
    "properties",
    "required",
    "nullable",
    "anyOf",
}


def _inline_refs(schema: dict, defs: dict | None = None) -> dict:
    """Resolve every `$ref` against `$defs` and drop the `$defs` table.

    Crucial for nested Pydantic models — Gemini's tool API doesn't follow
    `$ref`, so a `list[Reaction]` whose item schema is `{"$ref": "#/$defs/Reaction"}`
    becomes opaque to the model and it returns `null` for every array slot.
    """
    if not isinstance(schema, dict):
        return schema
    if defs is None:
        defs = schema.get("$defs") or schema.get("definitions") or {}

    if "$ref" in schema:
        ref = schema["$ref"]
        # Supports the standard `#/$defs/Name` and `#/definitions/Name` forms.
        for prefix in ("#/$defs/", "#/definitions/"):
            if ref.startswith(prefix):
                target = defs.get(ref[len(prefix):])
                if target is None:
                    return {}
                # Merge any sibling keys from the wrapping schema (e.g. `description`).
                merged = _inline_refs(target, defs)
                extras = {k: v for k, v in schema.items() if k != "$ref"}
                if extras:
                    merged = {**_inline_refs(extras, defs), **merged}
                return merged
        return schema  # unknown ref form — leave alone

    out: dict = {}
    for k, v in schema.items():
        if k in ("$defs", "definitions"):
            continue
        if isinstance(v, dict):
            out[k] = _inline_refs(v, defs)
        elif isinstance(v, list):
            out[k] = [_inline_refs(x, defs) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _sanitize_schema(schema: dict) -> dict:
    """Strip JSON-Schema features Gemini's tool API rejects, recursively."""
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    for k, v in schema.items():
        if k not in _GEMINI_SCHEMA_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _sanitize_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _sanitize_schema(v) if isinstance(v, dict) else v
        elif k == "anyOf" and isinstance(v, list):
            out[k] = [_sanitize_schema(x) for x in v]
        else:
            out[k] = v
    return out


def _build_function_declarations() -> list[dict]:
    """Translate Anthropic-style tool schemas into Gemini function declarations.

    Gemini-specific: every `$ref` is inlined before the strict-key sanitizer
    runs, otherwise nested Pydantic models (e.g. `list[Reaction]`) become
    opaque and the model returns `null` for every array slot.
    """
    fig_schema = _sanitize_schema(_inline_refs(FigureExtraction.model_json_schema()))
    decls: list[dict] = []
    for s in TOOL_SCHEMAS:
        if s["name"] == "submit_extraction":
            params = fig_schema
        else:
            params = _sanitize_schema(_inline_refs(copy.deepcopy(s["input_schema"])))
        decls.append(
            {
                "name": s["name"],
                "description": s["description"],
                "parameters": params,
            }
        )
    return decls


# ---------- previews (UI parity with the Anthropic extractor) ----------

def _format_args_preview(name: str, args: dict) -> str:
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
    result = result or {}
    if not result.get("ok", False):
        return f"error · {str(result.get('error') or 'error')[:80]}"
    if name == "validate_smiles":
        return f"ok · {result.get('molecular_formula', '?')} · {result.get('exact_mass', '?')}"
    if name == "submit_extraction":
        return "accepted"
    return "ok"


# ---------- main streaming entry point ----------

def extract_figure_stream_gemini_parallel(
    image_path: str | Path,
    *,
    model: str,
    client,
    system_prompt: str,
) -> Iterator[dict]:
    """Run the Gemini agent loop with parallel function calling enabled.

    Yields the same event dicts as the Anthropic / OpenAI extractors:
    step_start / step_done / complete / error."""
    from google.genai import types  # local import; only needed when Gemini is on

    image_path = Path(image_path)
    media_type, _ = mimetypes.guess_type(str(image_path))
    if media_type is None or not media_type.startswith("image/"):
        media_type = "image/png"

    function_decls = _build_function_declarations()
    tools = [types.Tool(function_declarations=function_decls)]

    user_text = (
        f"Extract every reaction in this figure (filename: {image_path.name}). "
        "Validate every SMILES — call `validate_smiles` for all of them in "
        "parallel in the same turn rather than one at a time, since the calls "
        "are independent. Once every SMILES is validated, call "
        "`submit_extraction` exactly once with the final result."
    )

    # Conversation history. Gemini uses Content/Part objects with `role` set
    # to "user" or "model"; tool responses go in user-role parts.
    history: list = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type=media_type),
                types.Part.from_text(text=user_text),
            ],
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools,
        # Force the model to call functions until submit_extraction terminates.
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY"),
        ),
        # We run the function-call → function-response loop ourselves below;
        # disabling the SDK's automatic loop is required when we want to see
        # every parallel call in one turn (otherwise the SDK may serialise
        # them into separate generate_content invocations on our behalf).
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True,
        ),
        max_output_tokens=8192,
    )

    submitted_dump: Optional[dict] = None
    tool_calls = 0
    input_tokens = 0
    output_tokens = 0
    last_step_index = 0

    try:
        for step in range(MAX_AGENT_STEPS):
            last_step_index = step
            resp = client.models.generate_content(
                model=model,
                contents=history,
                config=config,
            )
            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                input_tokens += getattr(usage, "prompt_token_count", 0) or 0
                output_tokens += getattr(usage, "candidates_token_count", 0) or 0

            candidate = (resp.candidates or [None])[0]
            if candidate is None or candidate.content is None:
                yield {"event": "error", "message": "Gemini returned no candidates"}
                return

            # Append the full candidate content (preserves any
            # thought_signature attached to its parts — required by Gemini 3
            # to keep tool-call context consistent across turns).
            history.append(candidate.content)

            # Collect every function call in this turn — Gemini 3 emits
            # several at once when the calls are independent (e.g. multiple
            # validate_smiles).
            calls = [p.function_call for p in (candidate.content.parts or []) if p.function_call]

            if not calls:
                # Model produced text without invoking a tool — agent is stuck.
                break

            response_parts = []
            for fc in calls:
                tool_calls += 1
                name = fc.name
                args = dict(fc.args) if fc.args else {}

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
                            "detail": e.errors()[:3],
                        }
                elif name in TOOL_DISPATCH:
                    try:
                        result = TOOL_DISPATCH[name](**args)
                    except Exception as exc:
                        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                else:
                    result = {"ok": False, "error": f"unknown tool {name}"}

                # Echo the call id back so Gemini 3 can map each response to
                # the right call when several were emitted in parallel.
                fc_id = getattr(fc, "id", None)
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=fc_id,
                            name=name,
                            response=result,
                        )
                    )
                )

                yield {
                    "event": "step_done",
                    "n": tool_calls,
                    "tool": name,
                    "ok": bool(result.get("ok")),
                    "summary": _format_result_summary(name, result),
                }

            history.append(types.Content(role="user", parts=response_parts))

            if submitted_dump is not None:
                break

        if submitted_dump is None:
            yield {
                "event": "error",
                "message": (
                    f"Agent did not call submit_extraction within "
                    f"{MAX_AGENT_STEPS} round-trips for {image_path.name}"
                ),
            }
            return

        metadata = {
            "image": str(image_path),
            "model": model,
            "provider": "gemini",
            "steps": last_step_index + 1,
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        yield {"event": "complete", "extraction": submitted_dump, "metadata": metadata}

    except Exception as exc:
        yield {"event": "error", "message": f"{type(exc).__name__}: {exc}"}
