"""Subprocess entry point — runs MolNexTR inference and prints JSON to stdout.

Lives in tools/ but is INVOKED via .venv-molnextr/bin/python (Python 3.10).
The main pipeline (Python 3.12) calls this through tools/molnextr_tool.py
because MolNexTR's deps (timm 0.4.12, OpenNMT-py 2.2.0, pyonmttok) don't
install on Python 3.12.

Usage:
    python tools/molnextr_runner.py --image PATH [--bbox X Y W H] [--checkpoint PATH] \
        [--device mps|cpu] [--return-atoms-bonds]

Output (single JSON object on stdout):
    {smiles, confidence, atoms, bonds, error}
"""
from __future__ import annotations

# --- _lzma shim (must come before any torch/torchvision/timm imports). --------
import sys
import types

if "_lzma" not in sys.modules:
    _m = types.ModuleType("_lzma")
    for _attr in (
        "FORMAT_XZ", "FORMAT_ALONE", "FORMAT_RAW", "FORMAT_AUTO",
        "CHECK_NONE", "CHECK_CRC32", "CHECK_CRC64", "CHECK_SHA256",
        "CHECK_ID_MAX", "CHECK_UNKNOWN",
        "FILTER_LZMA1", "FILTER_LZMA2", "FILTER_DELTA",
        "FILTER_X86", "FILTER_POWERPC", "FILTER_IA64",
        "FILTER_ARM", "FILTER_ARMTHUMB", "FILTER_SPARC",
        "MF_HC3", "MF_HC4", "MF_BT2", "MF_BT3", "MF_BT4",
        "MODE_FAST", "MODE_NORMAL",
        "PRESET_DEFAULT", "PRESET_EXTREME",
    ):
        setattr(_m, _attr, 0)

    class _Dummy:
        def __init__(self, *a, **k): pass
        def decompress(self, *a, **k): return b""
        def compress(self, *a, **k): return b""
        def flush(self, *a, **k): return b""

    _m.LZMADecompressor = _Dummy
    _m.LZMACompressor = _Dummy

    class _Err(Exception): pass
    _m.LZMAError = _Err
    _m.is_check_supported = lambda x: False
    _m._encode_filter_properties = lambda x: b""
    _m._decode_filter_properties = lambda x, y: {}
    sys.modules["_lzma"] = _m
# ------------------------------------------------------------------------------

import argparse
import json
import os
from pathlib import Path

# Make `import molnextr` resolve when run from anywhere.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _emit(obj: dict) -> None:
    """Write a single-line JSON to stdout (callers parse a single line)."""
    print(json.dumps(obj, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to source image.")
    parser.add_argument("--bbox", type=int, nargs=4, default=None,
                        metavar=("X", "Y", "W", "H"),
                        help="Optional crop region in pixels (top-left origin).")
    parser.add_argument("--pad", type=int, default=10, help="Padding around bbox.")
    parser.add_argument("--checkpoint", default=str(REPO / "molnextr" / "molnextr_official.pth"))
    parser.add_argument("--device", default=None, help="mps | cpu | cuda. Defaults to MPS if available.")
    parser.add_argument("--return-atoms-bonds", action="store_true")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        _emit({"error": f"image not found: {args.image}"})
        return 1
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        _emit({"error": f"checkpoint not found: {args.checkpoint}"})
        return 1

    try:
        import cv2
        import torch
    except Exception as e:
        _emit({"error": f"deps missing in molnextr venv: {type(e).__name__}: {e}"})
        return 1

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    img = cv2.imread(str(img_path))
    if img is None:
        _emit({"error": f"could not read image: {args.image}"})
        return 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if args.bbox is not None:
        x, y, w, h = args.bbox
        pad = max(0, args.pad)
        H, W, _ = img.shape
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(W, x + w + pad)
        bottom = min(H, y + h + pad)
        if right <= left or bottom <= top:
            _emit({"error": "empty crop region after clamping"})
            return 1
        img = img[top:bottom, left:right]

    # Quiet noisy module-level warnings from molnextr.
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    try:
        from molnextr import MolNexTR
    except Exception as e:
        _emit({"error": f"import MolNexTR failed: {type(e).__name__}: {e}"})
        return 1

    try:
        model = MolNexTR(str(ckpt_path), device=device)
    except Exception as e:
        _emit({"error": f"model load failed: {type(e).__name__}: {e}"})
        return 1

    # MolNexTR / chemistry.py prints debug info to stdout; redirect to stderr
    # so the runner's stdout stays a single clean JSON line.
    import contextlib, io
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            out = model.predict_image(
                img,
                return_atoms_bonds=args.return_atoms_bonds,
                return_confidence=True,
            )
    except Exception as e:
        sys.stderr.write(captured.getvalue())
        _emit({"error": f"inference failed: {type(e).__name__}: {e}"})
        return 1
    sys.stderr.write(captured.getvalue())

    result = {
        "smiles": out.get("smiles"),
        "confidence": float(out.get("confidence", 0.0)),
        "atoms_count": len(out.get("atoms", []) or []) or len(out.get("symbols", []) or []),
        "bonds_count": len(out.get("bonds", []) or []) or sum(
            sum(1 for v in row if v != 0)
            for row in (out.get("edges") or [])
        ) // 2,
        "device": str(device),
        "error": None,
    }
    if args.return_atoms_bonds:
        result["atoms"] = out.get("atoms")
        result["bonds"] = out.get("bonds")
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
