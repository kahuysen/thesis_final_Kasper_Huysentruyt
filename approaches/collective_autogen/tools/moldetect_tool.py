"""Agent-facing wrapper for MolDetect (rxnim.MolDetect) — molecule bbox detector.

Subprocesses to .venv-molnextr/bin/python (Python 3.10) because the deps
(rxnim, OpenNMT-py 2.2.0, transformers 4.47.0) only install on Python 3.10.

Result is disk-cached by image-mtime so repeat detection on the same image
is free.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .molnextr_tool import _TransientSubprocessError, _run_with_retry
from .trace import trace

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "tools" / "moldetect_runner.py"
DEFAULT_VENV_PY = REPO / ".venv-molnextr" / "bin" / "python"
DEFAULT_CHECKPOINT = REPO / "molnextr" / "moldet.ckpt"
CACHE_DIR = REPO / "cache" / "moldet"


def _cache_key(image_path: str, molecule_only: bool, score_min: float) -> str:
    p = Path(image_path).resolve()
    try:
        mtime = int(p.stat().st_mtime)
    except OSError:
        mtime = 0
    return hashlib.sha1(f"{p}|{mtime}|{molecule_only}|{score_min}".encode()).hexdigest()[:24]


@trace()
def detect_molecules(image_path: str, molecule_only: bool = True, score_min: float = 0.0) -> dict:
    """Detect molecule bounding boxes in an image.

    Args:
        image_path: path to the source image.
        molecule_only: if True (default), drops identifier / arrow / text bboxes
            and returns only structures of category '[Mol]'. Set False for the
            raw output.
        score_min: minimum score filter (rxnim's "score" is a token-id, not a
            probability — leave at 0 unless you know what you're doing).

    Returns:
        dict with keys:
            bboxes — list of {x1,y1,x2,y2,w,h,category,score,category_id} in
                     absolute pixel coords (top-left origin)
            image_w, image_h — image dimensions in pixels
            device — "mps" / "cuda" / "cpu"
            cached — bool, true if served from disk cache
            error  — None on success, a short string on failure
    """
    venv_py = Path(os.environ.get("MOLNEXTR_PYTHON", str(DEFAULT_VENV_PY)))
    checkpoint = Path(os.environ.get("MOLDET_CHECKPOINT", str(DEFAULT_CHECKPOINT)))

    if not Path(image_path).exists():
        return {"bboxes": [], "image_w": 0, "image_h": 0, "device": "",
                "cached": False, "error": f"image not found: {image_path}"}
    if not venv_py.exists():
        return {"bboxes": [], "image_w": 0, "image_h": 0, "device": "",
                "cached": False,
                "error": f"molnextr venv not found at {venv_py}; run requirements-molnextr.txt"}
    if not checkpoint.exists():
        return {"bboxes": [], "image_w": 0, "image_h": 0, "device": "",
                "cached": False,
                "error": f"checkpoint not found at {checkpoint}"}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(image_path, molecule_only, score_min)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            cached["cached"] = True
            return cached
        except Exception:
            pass

    cmd = [str(venv_py), str(RUNNER),
           "--image", str(Path(image_path).resolve()),
           "--checkpoint", str(checkpoint),
           "--score-min", str(score_min)]
    if molecule_only:
        cmd.append("--molecule-only")

    try:
        proc = _run_with_retry(cmd, timeout=240)
    except subprocess.TimeoutExpired:
        return {"bboxes": [], "image_w": 0, "image_h": 0, "device": "",
                "cached": False, "error": "subprocess timeout (240s, after retry)"}
    except _TransientSubprocessError as e:
        return {"bboxes": [], "image_w": 0, "image_h": 0, "device": "",
                "cached": False, "error": f"subprocess failed after retry: {e}"}
    except Exception as e:
        return {"bboxes": [], "image_w": 0, "image_h": 0, "device": "",
                "cached": False, "error": f"{type(e).__name__}: {e}"}

    out = (proc.stdout or "").strip().splitlines()
    if not out:
        return {"bboxes": [], "image_w": 0, "image_h": 0, "device": "",
                "cached": False,
                "error": f"empty stdout (rc={proc.returncode}); stderr: {proc.stderr[-300:] if proc.stderr else ''}"}
    try:
        result = json.loads(out[-1])
    except json.JSONDecodeError as e:
        return {"bboxes": [], "image_w": 0, "image_h": 0, "device": "",
                "cached": False, "error": f"could not parse runner output: {e}"}

    result.setdefault("cached", False)
    if not result.get("error"):
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False))
        except Exception:
            pass
    return result
