"""Per-run output directory bookkeeping.

Each upload gets a UUID and a directory layout:

    runs/<uuid>/
        input.<ext>          uploaded figure image
        extraction.json      validated FigureExtraction (on success)
        cards/*.png          rendered reaction cards
        reactions.csv        flat CSV
        insight.json         optional Rxn-INSIGHT enrichment
        insight.csv          optional Rxn-INSIGHT flat CSV
"""
from __future__ import annotations

import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def new_run() -> tuple[str, Path]:
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def run_path(run_id: str) -> Path | None:
    if not run_id or "/" in run_id or ".." in run_id:
        return None
    p = RUNS_DIR / run_id
    return p if p.is_dir() else None


def safe_child(run_dir: Path, name: str) -> Path | None:
    """Resolve `name` under `run_dir` while rejecting path traversal."""
    candidate = (run_dir / name).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.exists() else None


def find_input_image(run_dir: Path) -> Path | None:
    for p in run_dir.iterdir():
        if p.is_file() and p.name.startswith("input."):
            return p
    return None
