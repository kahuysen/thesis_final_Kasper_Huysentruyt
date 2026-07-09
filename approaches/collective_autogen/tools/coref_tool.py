"""Agent-facing wrapper for ChemEagle's label coreference detector.

`detect_label_coref` returns a list of bboxes (molecule structures + label
identifiers) and a set of (label_idx, struct_idx) pairs linking each
compound label ("1a", "2-Br", …) in the image to the structure it names.

Subprocesses to .venv-molnextr/bin/python (Python 3.10) — same venv as
MolNexTR / MolDetect. Result is disk-cached by image-mtime.

Default checkpoint location is ../ChemEagle/corefdet.ckpt (the user's
existing local install) so we don't duplicate the 393 MB file. Override
with COREFDET_CHECKPOINT.
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
RUNNER = REPO / "tools" / "coref_runner.py"
DEFAULT_VENV_PY = REPO / ".venv-molnextr" / "bin" / "python"
DEFAULT_CHECKPOINT = REPO.parent / "ChemEagle" / "corefdet.ckpt"
CACHE_DIR = REPO / "cache" / "coref"


def _cache_key(image_path: str, ocr: bool) -> str:
    p = Path(image_path).resolve()
    try:
        mtime = int(p.stat().st_mtime)
    except OSError:
        mtime = 0
    return hashlib.sha1(f"{p}|{mtime}|ocr={ocr}".encode()).hexdigest()[:24]


def _empty(error: str | None = None) -> dict:
    return {"bboxes": [], "corefs": [], "image_w": 0, "image_h": 0,
            "device": "", "cached": False, "error": error}


@trace()
def detect_label_coref(image_path: str, ocr: bool = True) -> dict:
    """Detect compound-label ↔ structure coreferences in a chemistry figure.

    Args:
        image_path: path to the source image.
        ocr: if True (default), run easyocr inside each detected label bbox
            to populate its `text` field (e.g. "1a", "2-Br"). Set False for
            ~30% faster runs when you only need the geometric pairing.

    Returns:
        dict with keys:
            bboxes  — list of {x1,y1,x2,y2,w,h, category, category_id, score, text}
                      in absolute pixel coords. category_id meaning:
                          1 = molecule structure
                          2 = generic text
                          3 = identifier/label
                          4 = superscript / footnote marker
            corefs  — list of [label_index, struct_index] pairs (indices into
                      `bboxes`). Use these to map a label to its drawn
                      structure deterministically.
            image_w, image_h — image dimensions in pixels
            device  — "mps" / "cuda" / "cpu"
            cached  — bool, true if served from disk cache
            error   — None on success, a short string on failure

    Use this when:
      - the figure is a substrate-scope grid with numbered labels (1a, 1b, …)
        that need to be linked to specific drawn structures, or
      - the same compound label is reused across multiple panels and you want
        to lock the label↔SMILES mapping before molecular_recognition runs.

    Skip when the figure is a single reaction with no compound numbering —
    coref adds latency without information in that case.
    """
    venv_py = Path(os.environ.get("MOLNEXTR_PYTHON", str(DEFAULT_VENV_PY)))
    checkpoint = Path(os.environ.get("COREFDET_CHECKPOINT", str(DEFAULT_CHECKPOINT)))

    if not Path(image_path).exists():
        return _empty(f"image not found: {image_path}")
    if not venv_py.exists():
        return _empty(f"molnextr venv not found at {venv_py}; "
                      f"run requirements-molnextr.txt to create it")
    if not checkpoint.exists():
        return _empty(f"corefdet checkpoint not found at {checkpoint}; "
                      f"set COREFDET_CHECKPOINT or place it at the default path")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(image_path, ocr)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            cached["cached"] = True
            return cached
        except Exception:
            pass

    cmd = [str(venv_py), str(RUNNER),
           "--image", str(Path(image_path).resolve()),
           "--checkpoint", str(checkpoint)]
    if not ocr:
        cmd.append("--no-ocr")

    try:
        # Coref + easyocr loads ~1.5 GB of weights; first run takes ~60-90s
        # cold, ~15s warm. Give it 5 minutes.
        proc = _run_with_retry(cmd, timeout=300)
    except subprocess.TimeoutExpired:
        return _empty("subprocess timeout (300s, after retry)")
    except _TransientSubprocessError as e:
        return _empty(f"subprocess failed after retry: {e}")
    except Exception as e:
        return _empty(f"{type(e).__name__}: {e}")

    out = (proc.stdout or "").strip().splitlines()
    if not out:
        return _empty(f"empty stdout (rc={proc.returncode}); "
                      f"stderr: {proc.stderr[-300:] if proc.stderr else ''}")
    try:
        result = json.loads(out[-1])
    except json.JSONDecodeError as e:
        return _empty(f"could not parse runner output: {e}")

    result.setdefault("cached", False)
    if not result.get("error"):
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False))
        except Exception:
            pass
    return result
