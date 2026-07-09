"""Subprocess entry — runs MolDetect(coref=True) and prints JSON to stdout.

Companion to tools/coref_tool.py. Invoked via .venv-molnextr/bin/python
(Python 3.10) because the deps (rxnim, easyocr, OpenNMT-py 2.2.0,
torch+torchvision pinned old) only install on Python 3.10.

Usage:
    python tools/coref_runner.py --image PATH \
        [--checkpoint PATH] [--device mps|cpu]

Output (single JSON line on stdout):
    {
      "bboxes": [
        {x1,y1,x2,y2,w,h, category, category_id, score, text}, ...
      ],
      "corefs": [[i_label, i_struct], ...],   # indices into "bboxes"
      "image_w", "image_h", "device", "error"
    }

Coordinate convention: absolute pixels in the ORIGINAL image (top-left
origin). rxnim resizes 3x internally for OCR but normalised bboxes are
scale-invariant, so we multiply by the original (W, H).
"""
from __future__ import annotations

# --- _lzma shim — must come BEFORE any torch / torchvision / timm imports. ----
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


def _norm_text(text_field) -> str:
    """easyocr returns a list of strings (or list of tuples); flatten to one str."""
    if text_field is None:
        return ""
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts = []
        for t in text_field:
            if isinstance(t, str):
                parts.append(t)
            elif isinstance(t, (list, tuple)) and len(t) >= 2 and isinstance(t[1], str):
                parts.append(t[1])
        return " ".join(p for p in parts if p)
    return str(text_field)


def _bbox_record(box: dict, image_w: int, image_h: int) -> dict:
    """Flatten one rxnim coref bbox dict into absolute pixel coords."""
    if not isinstance(box, dict):
        return {}
    raw = box.get("bbox") or box.get("box") or []
    if len(raw) != 4:
        return {}
    x1, y1, x2, y2 = (float(v) for v in raw)
    if max(x1, y1, x2, y2) <= 1.5:
        x1 *= image_w; x2 *= image_w
        y1 *= image_h; y2 *= image_h
    return {
        "x1": int(round(x1)), "y1": int(round(y1)),
        "x2": int(round(x2)), "y2": int(round(y2)),
        "w": int(round(x2 - x1)), "h": int(round(y2 - y1)),
        "category": box.get("category") or "",
        "category_id": int(box.get("category_id", 0)),
        "score": float(box.get("score", 0.0)),
        "text": _norm_text(box.get("text")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to corefdet.ckpt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip easyocr label-text reading (faster, but labels arrive empty).")
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

    # Skip MolDetect's eager loading of MolNexTR — the coref task only needs
    # bbox + label OCR, not SMILES (the molecular_recognition agent already
    # has its own MolNexTR call path). easyocr is loaded only when --no-ocr
    # is not set.
    try:
        import rxnim.interface as _ri
        _ri.MolDetect.get_molnextr = lambda self: None
        if args.no_ocr:
            _ri.MolDetect.get_ocr_model = lambda self: None
        from rxnim import MolDetect
    except Exception as e:
        _emit({"error": f"import MolDetect failed: {type(e).__name__}: {e}"}); return 1

    img = PILImage.open(img_path).convert("RGB")
    image_w, image_h = img.size

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            model = MolDetect(str(ckpt), device=device, coref=True)
            preds = model.predict_images(
                [img], batch_size=1,
                molnextr=False, coref=True, ocr=(not args.no_ocr),
            )
    except Exception as e:
        sys.stderr.write(captured.getvalue())
        _emit({"error": f"detection failed: {type(e).__name__}: {e}"}); return 1
    sys.stderr.write(captured.getvalue())

    if not preds:
        _emit({"bboxes": [], "corefs": [], "image_w": image_w,
               "image_h": image_h, "device": str(device), "error": None})
        return 0

    out = preds[0] or {}
    raw_bboxes = out.get("bboxes") or []
    raw_corefs = out.get("corefs") or []

    bboxes = []
    for b in raw_bboxes:
        rec = _bbox_record(b, image_w, image_h)
        if not rec or rec["w"] <= 0 or rec["h"] <= 0:
            # keep a placeholder so coref indices still align
            bboxes.append({"x1": 0, "y1": 0, "x2": 0, "y2": 0, "w": 0, "h": 0,
                           "category": "", "category_id": 0, "score": 0.0,
                           "text": "", "_dropped": True})
            continue
        bboxes.append(rec)

    corefs = []
    for pair in raw_corefs:
        try:
            i, j = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= i < len(bboxes) and 0 <= j < len(bboxes):
            corefs.append([i, j])

    _emit({
        "bboxes": bboxes,
        "corefs": corefs,
        "image_w": image_w,
        "image_h": image_h,
        "device": str(device),
        "error": None,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
