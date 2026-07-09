"""Image cropping tool — lets agents request a closer look at a region.

The function returns a path to the cropped PNG (under cache/crops/). main.py's
ImageRequestHandler picks up these paths from the tool trace and re-attaches
them as MultiModalMessages on the next selector turn.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from .trace import trace

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "crops"


@trace()
def crop_image_region(image_path: str, bbox: list[int], pad: int = 20) -> dict:
    """Crop a rectangular region from an image and write it to cache/crops/.

    Args:
        image_path: path to the source image.
        bbox: [x, y, w, h] in pixels (top-left origin). Coordinates are clamped
            to the image bounds.
        pad: extra pixels of padding around the bbox.

    Returns:
        {path: str|None, width: int, height: int, error: str|None}.
    """
    src = Path(image_path)
    if not src.exists():
        return {"path": None, "width": 0, "height": 0, "error": f"image not found: {image_path}"}
    try:
        img = Image.open(src).convert("RGB")
    except Exception as e:
        return {"path": None, "width": 0, "height": 0, "error": f"could not open image: {e}"}

    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return {"path": None, "width": 0, "height": 0, "error": "bbox must be [x, y, w, h]"}

    x, y, w, h = (int(v) for v in bbox)
    pad = max(0, int(pad))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(img.width, x + w + pad)
    bottom = min(img.height, y + h + pad)
    if right <= left or bottom <= top:
        return {"path": None, "width": 0, "height": 0, "error": "empty crop region after clamping"}

    cropped = img.crop((left, top, right, bottom))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{src.resolve()}|{left},{top},{right},{bottom}".encode()).hexdigest()[:16]
    out_path = CACHE_DIR / f"{src.stem}_{digest}.png"
    cropped.save(out_path)
    return {"path": str(out_path), "width": cropped.width, "height": cropped.height, "error": None}
