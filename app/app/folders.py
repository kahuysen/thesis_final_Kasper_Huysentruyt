"""Tiny folder store for the run picker.

Folder identity is decoupled from on-disk layout: every run still lives at
`runs/<uuid>/`. A single JSON sidecar at `runs/_meta.json` records folder
definitions and per-run folder assignments. Atomic writes + an in-process
lock cover the only realistic concurrency (browser tab + active extraction).

Schema:
    {
      "folders": [
        {"id": "<short-uuid>", "name": "Catalysts", "created": 1747...},
        ...
      ],
      "assignments": { "<run_id>": "<folder_id>" }   # only runs explicitly placed
    }

Runs not in `assignments` are "unfiled". Folder ids are short uuids.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .runs import RUNS_DIR

META_FILE = RUNS_DIR / "_meta.json"
_LOCK = threading.Lock()


def _read() -> dict:
    if not META_FILE.exists():
        return {"folders": [], "assignments": {}}
    try:
        raw = json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"folders": [], "assignments": {}}
    raw.setdefault("folders", [])
    raw.setdefault("assignments", {})
    return raw


def _write(data: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = META_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, META_FILE)


# ---------- public API ----------

def list_folders() -> list[dict]:
    with _LOCK:
        data = _read()
    return sorted(data["folders"], key=lambda f: f.get("created", 0))


def list_assignments() -> dict[str, str]:
    """run_id → folder_id for every explicitly-assigned run."""
    with _LOCK:
        return dict(_read()["assignments"])


def create_folder(name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("folder name cannot be empty")
    if len(name) > 80:
        raise ValueError("folder name too long")
    with _LOCK:
        data = _read()
        # Allow duplicate names (Finder-style); the id disambiguates.
        folder = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "created": time.time(),
        }
        data["folders"].append(folder)
        _write(data)
        return folder


def rename_folder(folder_id: str, name: str) -> Optional[dict]:
    name = (name or "").strip()
    if not name:
        raise ValueError("folder name cannot be empty")
    if len(name) > 80:
        raise ValueError("folder name too long")
    with _LOCK:
        data = _read()
        for f in data["folders"]:
            if f["id"] == folder_id:
                f["name"] = name
                _write(data)
                return f
        return None


def delete_folder(folder_id: str) -> str:
    """Delete an empty folder. Returns 'ok', 'not_found', or 'not_empty'."""
    with _LOCK:
        data = _read()
        idx = next((i for i, f in enumerate(data["folders"]) if f["id"] == folder_id), -1)
        if idx < 0:
            return "not_found"
        if any(v == folder_id for v in data["assignments"].values()):
            return "not_empty"
        data["folders"].pop(idx)
        _write(data)
        return "ok"


def assign_run(run_id: str, folder_id: Optional[str]) -> str:
    """Move a run to a folder, or to 'unfiled' when folder_id is None.

    Returns: 'ok', 'unknown_folder'.
    """
    with _LOCK:
        data = _read()
        if folder_id is None or folder_id == "":
            data["assignments"].pop(run_id, None)
            _write(data)
            return "ok"
        if not any(f["id"] == folder_id for f in data["folders"]):
            return "unknown_folder"
        data["assignments"][run_id] = folder_id
        _write(data)
        return "ok"


def folder_for_run(run_id: str) -> Optional[str]:
    with _LOCK:
        return _read()["assignments"].get(run_id)
