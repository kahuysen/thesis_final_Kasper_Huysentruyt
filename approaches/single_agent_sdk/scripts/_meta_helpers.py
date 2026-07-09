"""Shared helper for writing per-image cost/time sidecars.

Every benchmark_*.py runner calls `write_meta_sidecar(...)` after a
successful extraction. The sidecar lives next to the extraction JSON as
`<stem>.meta.json` and is consumed by scripts/summarize_costs.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_meta_sidecar(
    out_dir: Path,
    stem: str,
    *,
    metadata: dict[str, Any] | None,
    elapsed_s: float,
    tool_calls: int,
    backend: str,
    model: str,
) -> Path:
    """Write `<stem>.meta.json` capturing tokens, time, and round counts.

    Always writes the same shape so summarize_costs.py can aggregate without
    branching on backend.
    """
    metadata = metadata or {}
    meta = {
        "stem":           stem,
        "backend":        backend,
        "model":          model,
        "steps":          metadata.get("steps"),
        "tool_calls":     tool_calls,
        "input_tokens":   metadata.get("input_tokens"),
        "output_tokens":  metadata.get("output_tokens"),
        "elapsed_s":      round(float(elapsed_s), 2),
    }
    path = Path(out_dir) / f"{stem}.meta.json"
    path.write_text(json.dumps(meta, indent=2))
    return path
