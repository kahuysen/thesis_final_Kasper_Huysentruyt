"""Shim that exposes a Gemini (google-genai) client behind the OpenAI
`chat.completions.create(...)` surface used by the `_OS` agents, plus
tracing + token accounting.

Usage:
    client = get_client(base_url, api_key)
    # works for both CHEMEAGLE_BACKEND=gemini and default (OpenAI-compat Ollama/vLLM)

Env vars:
    CHEMEAGLE_BACKEND=gemini   -> use Gemini via google-genai
    GEMINI_API_KEY=...         -> required when backend=gemini

Supports:
  - messages with text parts and base64 `image_url` parts
  - system messages
  - response_format={"type": "json_object"}  -> Gemini response_mime_type
  - tools (OpenAI function format)           -> Gemini FunctionDeclaration
  - response.choices[0].message.content and .tool_calls[i].function.{name,arguments}
  - response.usage.{prompt_tokens,completion_tokens,total_tokens}

Tracing (opt-in via set_trace_dir(path)):
  - Every .create() call appends one JSON line to <trace_dir>/<image_key>.jsonl
  - set_image_context(key) / clear_image_context() scope a run to one image
  - get_image_usage() / get_global_usage() read token tallies
"""
from __future__ import annotations

import base64
import inspect
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

import openai


# --- public API -------------------------------------------------------------


def get_client(base_url: Optional[str] = None, api_key: Optional[str] = None):
    backend = os.getenv("CHEMEAGLE_BACKEND", "").strip().lower()
    if backend == "gemini":
        return GeminiOpenAIShim()
    return openai.OpenAI(base_url=base_url, api_key=api_key)


# --- tracing / usage state --------------------------------------------------

_lock = threading.Lock()
_ctx = threading.local()
_trace_dir: Optional[str] = None
_usage_totals = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "calls": 0,
}


def set_trace_dir(path: Optional[str]) -> None:
    """Where per-image JSONL traces are written. Pass None to disable."""
    global _trace_dir
    _trace_dir = path
    if path:
        os.makedirs(path, exist_ok=True)


def set_image_context(key: str) -> None:
    """Tag subsequent LLM calls on this thread as belonging to image <key>."""
    _ctx.image_key = key
    _ctx.local_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }


def clear_image_context() -> None:
    for attr in ("image_key", "local_usage"):
        if hasattr(_ctx, attr):
            delattr(_ctx, attr)


def get_image_usage() -> dict:
    return dict(
        getattr(
            _ctx,
            "local_usage",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
        )
    )


def get_global_usage() -> dict:
    with _lock:
        return dict(_usage_totals)


# --- shim internals ---------------------------------------------------------


class _Completions:
    def __init__(self, parent: "GeminiOpenAIShim"):
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._create(**kwargs)


class _Chat:
    def __init__(self, parent: "GeminiOpenAIShim"):
        self.completions = _Completions(parent)


class GeminiOpenAIShim:
    """Minimal OpenAI-compat surface backed by google-genai."""

    def __init__(self):
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "CHEMEAGLE_BACKEND=gemini requires GEMINI_API_KEY to be set"
            )
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.chat = _Chat(self)

    # --- explicit context-cache lifecycle (Gemini caches API) --------------

    def create_cache(
        self,
        *,
        model: str,
        system_instruction: Optional[str] = None,
        contents: Optional[list] = None,
        ttl_seconds: int = 3600,
        display_name: Optional[str] = None,
    ) -> Optional[str]:
        """Create an explicit context cache, return cache.name or None on failure.

        Use the returned name with `extra_body={"cached_content": name}` on
        subsequent chat.completions.create calls. Caller owns the cache and
        must call delete_cache(name) (e.g., in a try/finally) when done.
        """
        from google.genai import types

        if not system_instruction and not contents:
            return None
        cfg_kwargs: dict[str, Any] = {"ttl": f"{int(ttl_seconds)}s"}
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if contents:
            cfg_kwargs["contents"] = contents
        if display_name:
            cfg_kwargs["display_name"] = display_name
        try:
            cache = self._client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(**cfg_kwargs),
            )
        except Exception as e:
            _record_cache_event(
                action="create",
                model=model,
                name=None,
                ttl_seconds=ttl_seconds,
                error=f"{type(e).__name__}: {e}",
            )
            return None
        name = getattr(cache, "name", None)
        token_count = None
        try:
            token_count = int(getattr(cache, "usage_metadata", None).total_token_count)
        except Exception:
            pass
        _record_cache_event(
            action="create",
            model=model,
            name=name,
            ttl_seconds=ttl_seconds,
            token_count=token_count,
        )
        return name

    def delete_cache(self, name: Optional[str]) -> None:
        if not name:
            return
        try:
            self._client.caches.delete(name=name)
            _record_cache_event(action="delete", model=None, name=name)
        except Exception as e:
            _record_cache_event(
                action="delete", model=None, name=name,
                error=f"{type(e).__name__}: {e}",
            )

    def _create(
        self,
        *,
        model,
        messages,
        response_format=None,
        temperature=0,
        extra_body=None,
        tools=None,
        tool_choice=None,
        max_tokens=None,
        **_ignored,
    ):
        from google.genai import types

        system_chunks = []
        contents = []
        for msg in messages:
            role = _msg_get(msg, "role", "user")
            content = _msg_get(msg, "content")

            if role == "system":
                if isinstance(content, str):
                    system_chunks.append(content)
                elif isinstance(content, list):
                    system_chunks.extend(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                continue

            if role == "assistant":
                parts = []
                if content:
                    if isinstance(content, str):
                        parts.append(types.Part.from_text(text=content))
                    else:
                        parts.extend(_to_parts(content, types))
                for tc in _msg_get(msg, "tool_calls", None) or []:
                    raw_part = _msg_get(tc, "_raw_part", None)
                    if raw_part is not None:
                        # Preserves Gemini-3 thought_signature when echoing back.
                        parts.append(raw_part)
                        continue
                    fn = _msg_get(tc, "function", None)
                    fn_name = _msg_get(fn, "name", None) if fn is not None else None
                    if not fn_name:
                        continue
                    raw_args = _msg_get(fn, "arguments", None) if fn is not None else None
                    if isinstance(raw_args, str):
                        try:
                            fn_args = json.loads(raw_args) if raw_args else {}
                        except Exception:
                            fn_args = {"_raw_arguments": raw_args}
                    elif isinstance(raw_args, dict):
                        fn_args = raw_args
                    else:
                        fn_args = {}
                    parts.append(types.Part.from_function_call(name=fn_name, args=fn_args))
                if not parts:
                    parts = [types.Part.from_text(text="")]
                contents.append(types.Content(role="model", parts=parts))
                continue

            if role == "tool":
                tool_name = _msg_get(msg, "name", None) or "tool"
                if isinstance(content, str):
                    try:
                        resp_obj = json.loads(content)
                    except Exception:
                        resp_obj = {"content": content}
                else:
                    resp_obj = content
                if not isinstance(resp_obj, dict):
                    resp_obj = {"result": resp_obj}
                parts = [
                    types.Part.from_function_response(
                        name=tool_name, response=resp_obj
                    )
                ]
                contents.append(types.Content(role="user", parts=parts))
                continue

            parts = _to_parts(content, types)
            contents.append(types.Content(role="user", parts=parts))

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if system_chunks:
            config_kwargs["system_instruction"] = "\n".join(system_chunks)
        if response_format and response_format.get("type") == "json_object":
            config_kwargs["response_mime_type"] = "application/json"
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        else:
            config_kwargs["max_output_tokens"] = int(os.environ.get("CHEMEAGLE_MAX_OUTPUT_TOKENS", "32768"))
        # Optional thinking-budget knob (Gemini-3 only). Pass via
        # extra_body={"thinking_budget": 0} to disable thinking on simple
        # judging calls where the latency cost isn't justified.
        if isinstance(extra_body, dict) and "thinking_budget" in extra_body:
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=int(extra_body["thinking_budget"])
                )
            except Exception:
                pass
        # Optional explicit context cache (Gemini caches API). Pass via
        # extra_body={"cached_content": "cachedContents/abc..."}. When set,
        # the cached system_instruction/contents replace any inline ones,
        # so we drop system_instruction from this call's config to avoid
        # the API rejecting "both cache and system_instruction".
        if isinstance(extra_body, dict) and extra_body.get("cached_content"):
            config_kwargs["cached_content"] = extra_body["cached_content"]
            config_kwargs.pop("system_instruction", None)
        if tools:
            fn_decls = []
            for t in tools:
                fn = (t or {}).get("function", {})
                name = fn.get("name")
                if not name:
                    continue
                fn_decls.append(
                    types.FunctionDeclaration(
                        name=name,
                        description=fn.get("description", ""),
                        parameters=_clean_schema(
                            fn.get("parameters", {"type": "object"})
                        ),
                    )
                )
            if fn_decls:
                config_kwargs["tools"] = [
                    types.Tool(function_declarations=fn_decls)
                ]
                # Agents (e.g. get_R_group_sub_agent.py:2080) require at least
                # one tool call; force Gemini to call one of the declared fns.
                allowed = None
                if isinstance(tool_choice, dict):
                    fn_name = (
                        (tool_choice.get("function") or {}).get("name")
                        or tool_choice.get("name")
                    )
                    if fn_name:
                        allowed = [fn_name]
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        **({"allowed_function_names": allowed} if allowed else {}),
                    )
                )

        config = types.GenerateContentConfig(**config_kwargs)

        t0 = time.perf_counter()
        error_str = None
        response = None
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - t0
            _record_trace(
                model=model,
                messages=messages,
                tools=tools,
                content=None,
                tool_calls_public=[],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency=latency,
                error=error_str,
            )
            raise openai.APIError(str(e), request=None, body=None) from e

        latency = time.perf_counter() - t0

        # Extract usage
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0) if usage else 0

        # Tally
        with _lock:
            _usage_totals["prompt_tokens"] += prompt_tokens
            _usage_totals["completion_tokens"] += completion_tokens
            _usage_totals["total_tokens"] += total_tokens
            _usage_totals["calls"] += 1
        local = getattr(_ctx, "local_usage", None)
        if local is not None:
            local["prompt_tokens"] += prompt_tokens
            local["completion_tokens"] += completion_tokens
            local["total_tokens"] += total_tokens
            local["calls"] += 1

        shaped = _to_openai_response(response)
        shaped.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        # Public tool-call shape used by agents and in trace
        tool_calls_public = []
        try:
            for tc in shaped.choices[0].message.tool_calls or []:
                tool_calls_public.append(
                    {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )
        except Exception:
            pass

        _record_trace(
            model=model,
            messages=messages,
            tools=tools,
            content=shaped.choices[0].message.content,
            tool_calls_public=tool_calls_public,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency=latency,
            error=None,
        )
        return shaped


# --- helpers ---------------------------------------------------------------


def _msg_get(m, key, default=None):
    """Dict- or object-style accessor for OpenAI-shaped message items."""
    if isinstance(m, dict):
        return m.get(key, default)
    return getattr(m, key, default)


_AGENT_FILES = {
    "main.py",
    "get_text_agent.py",
    "get_molecular_agent.py",
    "get_reaction_agent.py",
    "get_R_group_sub_agent.py",
    "get_observer.py",
}


def _find_agent_caller() -> str:
    try:
        for fi in inspect.stack():
            base = os.path.basename(fi.filename)
            if base in _AGENT_FILES:
                return f"{base}:{fi.function}:{fi.lineno}"
    except Exception:
        pass
    return ""


def _extract_system(messages) -> str:
    chunks = []
    for m in messages or []:
        if _msg_get(m, "role") == "system":
            c = _msg_get(m, "content")
            if isinstance(c, str):
                chunks.append(c)
            elif isinstance(c, list):
                chunks.extend(
                    x.get("text", "")
                    for x in c
                    if isinstance(x, dict) and x.get("type") == "text"
                )
    return ("\n".join(chunks))[:4000]


def _extract_user_text_and_image_flag(messages):
    texts = []
    has_image = False
    for m in messages or []:
        if _msg_get(m, "role") == "system":
            continue
        c = _msg_get(m, "content")
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for x in c:
                if not isinstance(x, dict):
                    continue
                if x.get("type") == "text":
                    texts.append(x.get("text", ""))
                elif x.get("type") == "image_url":
                    has_image = True
    return ("\n".join(texts))[:8000], has_image


def _record_trace(
    *,
    model,
    messages,
    tools,
    content,
    tool_calls_public,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    latency,
    error,
):
    if not _trace_dir:
        return
    image_key = getattr(_ctx, "image_key", "_no_image_context")
    user_text, has_image = _extract_user_text_and_image_flag(messages)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "image": image_key,
        "caller": _find_agent_caller(),
        "model": model,
        "latency_s": round(latency, 3),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
        "system": _extract_system(messages),
        "user_text": user_text,
        "has_image": has_image,
        "tools_offered": [
            (t or {}).get("function", {}).get("name")
            for t in (tools or [])
            if (t or {}).get("function", {}).get("name")
        ],
        "response_content": content,
        "response_tool_calls": tool_calls_public,
        "error": error,
    }
    safe_key = "".join(c if (c.isalnum() or c in "._-") else "_" for c in image_key)
    path = os.path.join(_trace_dir, f"{safe_key}.jsonl")
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _record_cache_event(
    *,
    action: str,
    model: Optional[str],
    name: Optional[str],
    ttl_seconds: Optional[int] = None,
    token_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    if not _trace_dir:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": "cache",
        "action": action,
        "model": model,
        "name": name,
        "ttl_seconds": ttl_seconds,
        "token_count": token_count,
        "error": error,
    }
    path = os.path.join(_trace_dir, "_cache.jsonl")
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# --- part / schema / response translators -----------------------------------


def _to_parts(content, types):
    if content is None:
        return [types.Part.from_text(text="")]
    if isinstance(content, str):
        return [types.Part.from_text(text=content)]
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(types.Part.from_text(text=item))
            continue
        if not isinstance(item, dict):
            parts.append(types.Part.from_text(text=str(item)))
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(types.Part.from_text(text=item.get("text", "")))
        elif kind == "image_url":
            url = (item.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                header, _, b64 = url.partition(",")
                mime = "image/png"
                if ":" in header:
                    mime_part = header.split(":", 1)[1]
                    mime = mime_part.split(";", 1)[0] or mime
                try:
                    data = base64.b64decode(b64)
                    parts.append(
                        types.Part.from_bytes(data=data, mime_type=mime)
                    )
                except Exception:
                    parts.append(
                        types.Part.from_text(text=f"[could not decode image: {url[:40]}...]")
                    )
            else:
                parts.append(types.Part.from_text(text=f"[unsupported image url: {url[:60]}]"))
        else:
            parts.append(types.Part.from_text(text=json.dumps(item)))
    return parts or [types.Part.from_text(text="")]


def _clean_schema(schema):
    if not isinstance(schema, dict):
        return schema
    out = {}
    drop = {"additionalProperties", "$schema", "definitions", "$defs"}
    for k, v in schema.items():
        if k in drop:
            continue
        if isinstance(v, dict):
            out[k] = _clean_schema(v)
        elif isinstance(v, list):
            out[k] = [_clean_schema(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _to_openai_response(gem_response):
    text_chunks = []
    tool_calls = []
    try:
        cand = gem_response.candidates[0]
        for p in cand.content.parts or []:
            if getattr(p, "text", None):
                text_chunks.append(p.text)
            fc = getattr(p, "function_call", None)
            if fc and getattr(fc, "name", None):
                args = {}
                try:
                    args = dict(fc.args) if fc.args else {}
                except Exception:
                    args = {}
                tool_calls.append(
                    SimpleNamespace(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        type="function",
                        function=SimpleNamespace(
                            name=fc.name,
                            arguments=json.dumps(args, ensure_ascii=False),
                        ),
                        # Keep the raw Gemini Part so we can echo it back verbatim
                        # (preserves thought_signature required by Gemini 3).
                        _raw_part=p,
                    )
                )
    except Exception:
        txt = getattr(gem_response, "text", None)
        if txt:
            text_chunks.append(txt)

    content = "".join(text_chunks) if text_chunks else None
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=(tool_calls or None),
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice])
