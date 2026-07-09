"""Agent-facing wrapper for MolNexTR.

MolNexTR's deps (timm 0.4.12, OpenNMT-py 2.2.0, pyonmttok) only install on
Python 3.10. Our main pipeline runs on Python 3.12, so we shell out to a
side venv (.venv-molnextr) via subprocess and parse its JSON output.

Result of every call is also disk-cached by (image_path mtime, bbox) so
re-asking for the same crop is free.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from .trace import trace


class _TransientSubprocessError(RuntimeError):
    """Raised when a subprocess call looks recoverable (timeout, empty stdout)."""


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((subprocess.TimeoutExpired, _TransientSubprocessError)),
    reraise=True,
)
def _run_with_retry(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 and not (proc.stdout or "").strip():
        # Treat empty-stdout-with-nonzero-rc as transient (env hiccup, OOM, etc).
        raise _TransientSubprocessError(
            f"rc={proc.returncode}; stderr: {proc.stderr[-300:] if proc.stderr else ''}"
        )
    return proc

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "tools" / "molnextr_runner.py"
DEFAULT_VENV_PY = REPO / ".venv-molnextr" / "bin" / "python"
DEFAULT_CHECKPOINT = REPO / "molnextr" / "molnextr_official.pth"
CACHE_DIR = REPO / "cache" / "molnextr"


def _cache_key(image_path: str, bbox: list[int] | None, pad: int) -> str:
    p = Path(image_path).resolve()
    try:
        mtime = int(p.stat().st_mtime)
    except OSError:
        mtime = 0
    parts = f"{p}|{mtime}|{bbox}|{pad}"
    return hashlib.sha1(parts.encode()).hexdigest()[:24]


@trace()
def predict_molecule_smiles(
    image_path: str,
    bbox: list[int] | None = None,
    pad: int = 10,
) -> dict:
    """Run MolNexTR on an image (or a cropped region) and return the SMILES.

    Args:
        image_path: path to the source image.
        bbox: optional `[x, y, w, h]` in pixels (top-left origin). When the
            image contains multiple molecules, pass the bbox of a single
            structure for best results — MolNexTR is trained on
            single-molecule images.
        pad: extra pixels of padding around the bbox (default 10).

    Returns:
        dict with keys:
            smiles        — string SMILES (may include `*` / `[1*]` wildcards)
            confidence    — float 0..1; below ~0.3 the prediction is unreliable
            atoms_count   — number of atoms in the predicted graph
            bonds_count   — number of bonds in the predicted graph
            device        — "mps" / "cuda" / "cpu" used for inference
            cached        — bool, true if served from disk cache
            error         — None on success, a short string on failure
    """
    venv_py = Path(os.environ.get("MOLNEXTR_PYTHON", str(DEFAULT_VENV_PY)))
    checkpoint = Path(os.environ.get("MOLNEXTR_CHECKPOINT", str(DEFAULT_CHECKPOINT)))

    if not Path(image_path).exists():
        return {"smiles": None, "confidence": 0.0, "atoms_count": 0, "bonds_count": 0,
                "device": "", "cached": False, "error": f"image not found: {image_path}"}
    if not venv_py.exists():
        return {"smiles": None, "confidence": 0.0, "atoms_count": 0, "bonds_count": 0,
                "device": "", "cached": False,
                "error": f"molnextr venv not found at {venv_py}; create it via requirements-molnextr.txt"}
    if not checkpoint.exists():
        return {"smiles": None, "confidence": 0.0, "atoms_count": 0, "bonds_count": 0,
                "device": "", "cached": False,
                "error": f"checkpoint not found at {checkpoint}"}

    # Disk cache.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(image_path, bbox, pad)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            cached["cached"] = True
            return cached
        except Exception:
            pass  # fall through to live call

    cmd = [str(venv_py), str(RUNNER), "--image", str(Path(image_path).resolve()),
           "--checkpoint", str(checkpoint)]
    if bbox is not None:
        cmd += ["--bbox", *[str(int(v)) for v in bbox], "--pad", str(int(pad))]

    try:
        proc = _run_with_retry(cmd, timeout=180)
    except subprocess.TimeoutExpired:
        return {"smiles": None, "confidence": 0.0, "atoms_count": 0, "bonds_count": 0,
                "device": "", "cached": False, "error": "subprocess timeout (180s, after retry)"}
    except _TransientSubprocessError as e:
        return {"smiles": None, "confidence": 0.0, "atoms_count": 0, "bonds_count": 0,
                "device": "", "cached": False, "error": f"subprocess failed after retry: {e}"}
    except Exception as e:
        return {"smiles": None, "confidence": 0.0, "atoms_count": 0, "bonds_count": 0,
                "device": "", "cached": False, "error": f"{type(e).__name__}: {e}"}

    out = (proc.stdout or "").strip().splitlines()
    if not out:
        return {"smiles": None, "confidence": 0.0, "atoms_count": 0, "bonds_count": 0,
                "device": "", "cached": False,
                "error": f"empty stdout (rc={proc.returncode}); stderr: {proc.stderr[-300:] if proc.stderr else ''}"}
    try:
        result = json.loads(out[-1])
    except json.JSONDecodeError as e:
        return {"smiles": None, "confidence": 0.0, "atoms_count": 0, "bonds_count": 0,
                "device": "", "cached": False, "error": f"could not parse runner output: {e}"}

    result.setdefault("cached", False)
    if not result.get("error") and result.get("smiles"):
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False))
        except Exception:
            pass
    return result
