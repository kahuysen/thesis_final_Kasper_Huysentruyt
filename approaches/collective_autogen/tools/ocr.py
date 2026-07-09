"""Agent-facing wrapper for Tesseract OCR via pytesseract.

The text_extraction agent calls this BEFORE re-reading the image with the
vision LLM. Tesseract handles 80%+ of caption / footnote / heading text in
chemistry figures at near-zero cost; the LLM is then only needed when the
output is sparse, garbled, or contains chemistry symbols Tesseract doesn't
know.

Result is disk-cached by image-mtime so repeat reads on the same figure
during a multi-pass run are free.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytesseract
from PIL import Image

from .trace import trace

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "cache" / "ocr"


def _cache_key(image_path: str, lang: str, psm: int) -> str:
    p = Path(image_path).resolve()
    try:
        mtime = int(p.stat().st_mtime)
    except OSError:
        mtime = 0
    return hashlib.sha1(f"{p}|{mtime}|{lang}|{psm}".encode()).hexdigest()[:24]


@trace()
def ocr_image(image_path: str, lang: str = "eng", psm: int = 6) -> dict:
    """Run Tesseract OCR on a chemistry image and return raw text.

    Args:
        image_path: path to the image (PNG/JPG).
        lang: Tesseract language code (default "eng").
        psm: Tesseract page-segmentation mode. 6 = "assume a single uniform
            block of text" — the safe default for figure captions and
            tables. Try psm=11 ("sparse text") if the image is mostly
            structures with scattered labels.

    Returns:
        dict with keys:
            raw_text   — full OCR string with original line breaks
            lines      — list of non-empty stripped lines
            char_count — len(raw_text)
            cached     — bool
            error      — None on success, a short string on failure

    USE THIS BEFORE re-reading the image with vision. Vision is only needed
    when raw_text is empty, garbled, or chemistry-specific (e.g. R-group
    tables with subscripts Tesseract miscodes).
    """
    if not Path(image_path).exists():
        return {"raw_text": "", "lines": [], "char_count": 0,
                "cached": False, "error": f"image not found: {image_path}"}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(image_path, lang, psm)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            cached["cached"] = True
            return cached
        except Exception:
            pass

    try:
        img = Image.open(image_path)
        config = f"--psm {int(psm)}"
        raw_text = pytesseract.image_to_string(img, lang=lang, config=config)
    except pytesseract.TesseractNotFoundError as e:
        return {"raw_text": "", "lines": [], "char_count": 0,
                "cached": False,
                "error": f"tesseract binary not found: {e}; install via `brew install tesseract`"}
    except Exception as e:
        return {"raw_text": "", "lines": [], "char_count": 0,
                "cached": False, "error": f"{type(e).__name__}: {e}"}

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    result = {
        "raw_text": raw_text,
        "lines": lines,
        "char_count": len(raw_text),
        "cached": False,
        "error": None,
    }
    try:
        cache_file.write_text(json.dumps(result, ensure_ascii=False))
    except Exception:
        pass
    return result
