"""Tool-call tracing.

main.py sets the trace path before the team runs; every wrapped tool call
appends one JSON line.
"""
from __future__ import annotations

import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

_trace_path: Path | None = None


def set_trace_path(path: Path | None) -> None:
    global _trace_path
    _trace_path = path


def trace(agent_hint: str | None = None) -> Callable:
    """Wrap a callable so each invocation appends to the active trace file.

    `agent_hint` is informational — AutoGen does not pass the calling agent's
    name into tool functions, so the hint is best-effort metadata about which
    agent has the tool registered.
    """

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            err: str | None = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:  # never crash the agent loop
                err = f"{type(e).__name__}: {e}"
                return {"error": err}
            finally:
                if _trace_path is not None:
                    record = {
                        "ts": t0,
                        "duration_s": round(time.time() - t0, 4),
                        "tool": fn.__name__,
                        "agent_hint": agent_hint,
                        "args": _safe(args),
                        "kwargs": _safe(kwargs),
                        "error": err,
                    }
                    try:
                        with _trace_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except Exception:
                        pass  # tracing must never break the run

        return wrapper

    return deco


def _safe(x: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "<truncated>"
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_safe(v, depth + 1) for v in x]
    if isinstance(x, dict):
        return {str(k): _safe(v, depth + 1) for k, v in x.items()}
    return str(x)
