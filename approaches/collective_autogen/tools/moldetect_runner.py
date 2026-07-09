"""Subprocess entry — runs MolDetect (rxnim.MolDetect) and prints JSON bboxes.

Invoked via .venv-molnextr/bin/python (Python 3.10) from the main pipeline.
Companion to tools/moldetect_tool.py.

Usage:
    python tools/moldetect_runner.py --image PATH \
        [--checkpoint PATH] [--device mps|cpu]

Output (single JSON line on stdout):
    {bboxes: [{x1,y1,x2,y2,category,score}, ...], image_w, image_h, device, error}
"""
from __future__ import annotations

# --- _lzma shim — must come BEFORE any torch/torchvision/timm imports. -------
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
import contextlib
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False))


_MOL_CATEGORIES = {"[Mol]", "Mol", "molecule", "[mol]"}


def _bbox_record(box, image_w: int, image_h: int) -> dict:
    """Normalise rxnim's bbox output into a flat dict in absolute pixel coords.

    rxnim returns dicts like {category: '[Mol]', bbox: (x1,y1,x2,y2),
    category_id, score} where bbox values are NORMALISED to [0, 1].
    """
    if not isinstance(box, dict):
        return {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "category": "",
                "score": 0.0, "category_id": 0, "error": "non-dict record"}
    b = box.get("bbox") or box.get("box") or []
    cat = box.get("category") or box.get("label") or ""
    score = float(box.get("score", 0.0))
    cid = int(box.get("category_id", 0))
    if len(b) != 4:
        return {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "category": cat,
                "score": score, "category_id": cid, "error": "malformed bbox"}
    x1, y1, x2, y2 = (float(v) for v in b)
    # Detect normalised vs absolute coordinates: rxnim returns normalised.
    if max(x1, y1, x2, y2) <= 1.5:
        x1 *= image_w; x2 *= image_w
        y1 *= image_h; y2 *= image_h
    return {
        "x1": int(round(x1)), "y1": int(round(y1)),
        "x2": int(round(x2)), "y2": int(round(y2)),
        "w": int(round(x2 - x1)), "h": int(round(y2 - y1)),
        "category": cat, "score": score, "category_id": cid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=str(REPO / "molnextr" / "moldet.ckpt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-min", type=float, default=0.0)
    parser.add_argument("--molecule-only", action="store_true",
                        help="Filter results to molecule bboxes only (drop arrows / text).")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        _emit({"error": f"image not found: {args.image}"}); return 1
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        _emit({"error": f"checkpoint not found: {args.checkpoint}"}); return 1

    try:
        import torch
        from PIL import Image as PILImage
    except Exception as e:
        _emit({"error": f"deps missing: {type(e).__name__}: {e}"}); return 1

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Skip MolDetect's eager loading of MolNexTR + easyocr — we just want bboxes.
    # rxnim.interface.MolDetect.__init__ assigns these to self regardless of whether
    # predict_images uses them; replacing the loaders with no-ops avoids loading
    # ~1.5GB of unused weights.
    try:
        import rxnim.interface as _ri
        _ri.MolDetect.get_ocr_model = lambda self: None
        _ri.MolDetect.get_molnextr = lambda self: None
        from rxnim import MolDetect
    except Exception as e:
        _emit({"error": f"import MolDetect failed: {type(e).__name__}: {e}"}); return 1

    img = PILImage.open(img_path).convert("RGB")
    image_w, image_h = img.size

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            model = MolDetect(str(ckpt), device=device)
            preds = model.predict_images([img], batch_size=1, molnextr=False, ocr=False)
    except Exception as e:
        sys.stderr.write(captured.getvalue())
        _emit({"error": f"detection failed: {type(e).__name__}: {e}"}); return 1
    sys.stderr.write(captured.getvalue())

    bboxes = []
    for box in (preds[0] if preds else []) or []:
        rec = _bbox_record(box, image_w, image_h)
        if rec.get("error"):
            continue
        if rec["w"] <= 0 or rec["h"] <= 0:
            continue
        if rec["score"] < args.score_min:
            continue
        if args.molecule_only and rec["category"] not in _MOL_CATEGORIES:
            continue
        bboxes.append(rec)

    _emit({
        "bboxes": bboxes,
        "image_w": image_w,
        "image_h": image_h,
        "device": str(device),
        "error": None,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
