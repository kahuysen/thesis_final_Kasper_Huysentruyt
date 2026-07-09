"""Draw MolDetect bboxes onto each benchmark image.

Reads detections via tools.moldetect_tool.detect_molecules (cache-backed),
saves annotated copies to eval/benchmark/images_annotated/.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.moldetect_tool import detect_molecules

REPO = Path(__file__).resolve().parent
SRC = REPO / "eval" / "benchmark" / "images"
DST = REPO / "eval" / "benchmark" / "images_annotated"

CATEGORY_COLORS = {
    "[Mol]": "#e6194b",
    "[Txt]": "#3cb44b",
    "[Idt]": "#4363d8",
    "[Sup]": "#f58231",
    "[Rxn]": "#911eb4",
}
DEFAULT_COLOR = "#ffe119"


def _font(size: int = 14) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(image_path: Path, dst_path: Path, molecule_only: bool = True) -> dict:
    res = detect_molecules(str(image_path), molecule_only=molecule_only)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(max(12, img.height // 50))

    for i, b in enumerate(res.get("bboxes", [])):
        cat = b.get("category", "?")
        color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{i}:{cat}"
        tx, ty = x1 + 2, max(0, y1 - font.size - 2)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((tx, ty), label, fill="white", font=font)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path)
    return res


def main() -> None:
    images = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        print(f"no images found in {SRC}")
        return
    for p in images:
        dst = DST / f"{p.stem}_bboxes{p.suffix}"
        res = annotate(p, dst)
        n = len(res.get("bboxes", []))
        err = res.get("error")
        tag = "cache" if res.get("cached") else "fresh"
        suffix = f" error={err}" if err else ""
        print(f"{p.name}: {n} bboxes ({tag}) -> {dst.relative_to(REPO)}{suffix}")


if __name__ == "__main__":
    main()
