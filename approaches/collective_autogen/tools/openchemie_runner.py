"""Subprocess entry — runs OpenChemIE.extract_reactions_from_figures on a list
of images and prints JSON to stdout.

Invoked from .venv-openchemie/bin/python (Python 3.10) because OpenChemIE pins
torch<2.0 / opencv 4.5.5.64 etc., which conflict with our main venv (3.12) and
the molnextr venv (3.10 with torch 2.x).

Usage:
    python tools/openchemie_runner.py --images p1 p2 ... [--device cpu|mps|cuda]
                                      [--no-coref]

Output (one JSON object per stdout line, plus a trailing summary line):
    [{"image": "...", "reactions": [...], "error": null}, ...]
"""
from __future__ import annotations

# --- _lzma shim (must come before any torch/torchvision/timm imports). --------
# pyenv-installed Python 3.10 is built without xz/lzma support; torchvision >= 0.13
# imports lzma transitively. Same workaround as tools/molnextr_runner.py.
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
from pathlib import Path


def _emit(obj: dict | list) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--device", default=None,
                        help="cpu | mps | cuda. Default: cpu (avoids contention with molnextr).")
    parser.add_argument("--molscribe-coref", action="store_true",
                        help="Pass molscribe_coref=True (default off — slower, more reagents).")
    args = parser.parse_args()

    try:
        import torch
        from openchemie import OpenChemIE
    except Exception as e:
        _emit({"error": f"import failed: {type(e).__name__}: {e}"})
        return 1

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    try:
        model = OpenChemIE(device=device)
    except TypeError:
        # Some versions don't accept `device` kwarg.
        model = OpenChemIE()
    except Exception as e:
        _emit({"error": f"model init failed: {type(e).__name__}: {e}"})
        return 1

    try:
        from PIL import Image
    except Exception as e:
        _emit({"error": f"PIL import failed: {e}"})
        return 1

    images: list = []
    image_paths: list[str] = []
    for p in args.images:
        if not Path(p).exists():
            _emit({"error": f"image not found: {p}"})
            return 1
        images.append(Image.open(p).convert("RGB"))
        image_paths.append(p)

    try:
        results = model.extract_reactions_from_figures(
            images, molscribe_coref=args.molscribe_coref
        )
    except TypeError:
        # Older API: no kwarg.
        results = model.extract_reactions_from_figures(images)
    except Exception as e:
        _emit({"error": f"extract_reactions_from_figures failed: {type(e).__name__}: {e}"})
        return 1

    out = []
    for img_path, rec in zip(image_paths, results):
        out.append({
            "image": img_path,
            "reactions": rec.get("reactions", []) if isinstance(rec, dict) else rec,
            "error": None,
        })
    _emit(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
