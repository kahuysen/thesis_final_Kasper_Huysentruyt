"""Token-usage capture helpers shared by suite runners.

Each autogen `BaseChatMessage` carries `models_usage: RequestUsage | None`
with `prompt_tokens` / `completion_tokens` ints (set on messages produced by
a model, None on user/tool-result messages). We sum these post-hoc so the
runners don't need to instrument every model client call site.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def summarise_autogen_messages(messages: Iterable, model: str | None) -> dict:
    """Walk autogen messages and return a token-usage breakdown.

    Returns:
        {
          "model": <model name passed in>,
          "prompt_tokens": int,
          "completion_tokens": int,
          "total_tokens": int,
          "calls": int,            # how many messages carried a non-zero usage
          "by_agent": {
             "<source>": {"prompt_tokens": int, "completion_tokens": int, "calls": int},
             ...
          },
        }
    """
    by_agent: dict[str, dict[str, int]] = {}
    p_total = 0
    c_total = 0
    calls = 0
    for m in messages:
        u = getattr(m, "models_usage", None)
        if u is None:
            continue
        p = int(getattr(u, "prompt_tokens", 0) or 0)
        c = int(getattr(u, "completion_tokens", 0) or 0)
        if p == 0 and c == 0:
            continue
        src = getattr(m, "source", None) or "unknown"
        agent = by_agent.setdefault(
            src, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        )
        agent["prompt_tokens"] += p
        agent["completion_tokens"] += c
        agent["calls"] += 1
        p_total += p
        c_total += c
        calls += 1
    return {
        "model": model,
        "prompt_tokens": p_total,
        "completion_tokens": c_total,
        "total_tokens": p_total + c_total,
        "calls": calls,
        "by_agent": by_agent,
    }


def write_usage_json(path: Path, usage: dict) -> None:
    path.write_text(json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8")


def load_usage_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def empty_usage(model: str | None = None) -> dict:
    return {
        "model": model,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "by_agent": {},
    }


def add_usage(a: dict, b: dict) -> dict:
    """Sum two usage dicts. Per-agent entries are merged by source name."""
    out = {
        "model": a.get("model") or b.get("model"),
        "prompt_tokens": int(a.get("prompt_tokens", 0)) + int(b.get("prompt_tokens", 0)),
        "completion_tokens": int(a.get("completion_tokens", 0)) + int(b.get("completion_tokens", 0)),
        "total_tokens": int(a.get("total_tokens", 0)) + int(b.get("total_tokens", 0)),
        "calls": int(a.get("calls", 0)) + int(b.get("calls", 0)),
        "by_agent": {},
    }
    for src, vals in (a.get("by_agent") or {}).items():
        out["by_agent"][src] = dict(vals)
    for src, vals in (b.get("by_agent") or {}).items():
        cur = out["by_agent"].setdefault(
            src, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        )
        cur["prompt_tokens"] += int(vals.get("prompt_tokens", 0))
        cur["completion_tokens"] += int(vals.get("completion_tokens", 0))
        cur["calls"] += int(vals.get("calls", 0))
    return out
