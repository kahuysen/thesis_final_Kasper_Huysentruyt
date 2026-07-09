"""Python-3.12-side wrapper for OpenChemIE.

Subprocesses to .venv-openchemie/bin/python because OpenChemIE pins torch<2.0
and assorted opencv/Pillow versions that don't coexist with the main venv.

Disk-cached by image mtime so the same call is free on re-run.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "tools" / "openchemie_runner.py"
DEFAULT_VENV_PY = REPO / ".venv-openchemie" / "bin" / "python"
CACHE_DIR = REPO / "cache" / "openchemie"


def _cache_key(image_path: str, molscribe_coref: bool, device: str) -> str:
    p = Path(image_path).resolve()
    try:
        mtime = int(p.stat().st_mtime)
    except OSError:
        mtime = 0
    return hashlib.sha1(f"{p}|{mtime}|{molscribe_coref}|{device}".encode()).hexdigest()[:24]


def extract_reactions(
    image_path: str,
    *,
    molscribe_coref: bool = False,
    device: str = "cpu",
    timeout: int = 600,
) -> dict:
    """Run OpenChemIE on a single image; return {image, reactions, error, cached}.

    `device` defaults to "cpu" so this can run in parallel with molnextr (which
    uses MPS) without contending for GPU memory.
    """
    venv_py = Path(os.environ.get("OPENCHEMIE_PYTHON", str(DEFAULT_VENV_PY)))
    if not Path(image_path).exists():
        return {"image": image_path, "reactions": [], "cached": False,
                "error": f"image not found: {image_path}"}
    if not venv_py.exists():
        return {"image": image_path, "reactions": [], "cached": False,
                "error": f"openchemie venv not found at {venv_py}"}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(image_path, molscribe_coref, device)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            cached["cached"] = True
            return cached
        except Exception:
            pass

    cmd = [str(venv_py), str(RUNNER),
           "--images", str(Path(image_path).resolve()),
           "--device", device]
    if molscribe_coref:
        cmd.append("--molscribe-coref")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"image": image_path, "reactions": [], "cached": False,
                "error": f"subprocess timeout ({timeout}s)"}
    except Exception as e:
        return {"image": image_path, "reactions": [], "cached": False,
                "error": f"{type(e).__name__}: {e}"}

    out_lines = [ln for ln in (proc.stdout or "").strip().splitlines() if ln.strip()]
    if not out_lines:
        return {"image": image_path, "reactions": [], "cached": False,
                "error": f"empty stdout (rc={proc.returncode}); stderr: {proc.stderr[-300:] if proc.stderr else ''}"}
    try:
        parsed = json.loads(out_lines[-1])
    except json.JSONDecodeError as e:
        return {"image": image_path, "reactions": [], "cached": False,
                "error": f"could not parse runner output: {e}"}

    if isinstance(parsed, dict) and parsed.get("error"):
        return {"image": image_path, "reactions": [], "cached": False, "error": parsed["error"]}

    # parsed is a list-of-1 from the runner.
    record = parsed[0] if isinstance(parsed, list) and parsed else parsed
    record.setdefault("cached", False)
    record.setdefault("error", None)

    if not record.get("error"):
        try:
            cache_file.write_text(json.dumps(record, ensure_ascii=False))
        except Exception:
            pass
    return record
