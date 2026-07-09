"""Dump the per-bbox crops MolNexTR actually sees, plus its predictions.

Replicates tools/molnextr_runner.py's crop math (bbox + pad=10, clamped to
image bounds) so each saved PNG matches what the model receives at inference.
Pair this with eval/benchmark/images_annotated/ to inspect the full pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.moldetect_tool import detect_molecules
from tools.molnextr_tool import predict_molecule_smiles

REPO = Path(__file__).resolve().parent
SRC = REPO / "eval" / "benchmark" / "images"
DST = REPO / "eval" / "benchmark" / "molnextr_crops"

PAD = 10  # matches molnextr_runner.py default


def _crop(img: Image.Image, x: int, y: int, w: int, h: int, pad: int) -> Image.Image:
    W, H = img.size
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(W, x + w + pad)
    bottom = min(H, y + h + pad)
    return img.crop((left, top, right, bottom))


def process(image_path: Path) -> dict:
    det = detect_molecules(str(image_path), molecule_only=True)
    bboxes = det.get("bboxes", []) or []

    out_dir = DST / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    entries = []
    for i, b in enumerate(bboxes):
        x, y, w, h = int(b["x1"]), int(b["y1"]), int(b["w"]), int(b["h"])
        crop = _crop(img, x, y, w, h, PAD)
        crop_path = out_dir / f"bbox_{i:02d}.png"
        crop.save(crop_path)

        pred = predict_molecule_smiles(str(image_path), bbox=[x, y, w, h], pad=PAD)
        entries.append({
            "index": i,
            "bbox_xywh": [x, y, w, h],
            "bbox_xyxy": [int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])],
            "category": b.get("category"),
            "score": b.get("score"),
            "crop_file": crop_path.name,
            "crop_size": list(crop.size),
            "smiles": pred.get("smiles"),
            "confidence": pred.get("confidence"),
            "molnextr_cached": pred.get("cached", False),
            "error": pred.get("error"),
        })

    sidecar = {
        "source_image": image_path.name,
        "image_size": list(img.size),
        "pad": PAD,
        "moldetect_cached": det.get("cached", False),
        "moldetect_error": det.get("error"),
        "n_bboxes": len(bboxes),
        "entries": entries,
    }
    (out_dir / "predictions.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return sidecar


def main() -> None:
    images = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        print(f"no images found in {SRC}")
        return
    for p in images:
        s = process(p)
        n_ok = sum(1 for e in s["entries"] if e["smiles"] and not e["error"])
        print(f"{p.name}: {s['n_bboxes']} crops -> "
              f"{(DST / p.stem).relative_to(REPO)}/  "
              f"({n_ok}/{s['n_bboxes']} smiles ok)")


if __name__ == "__main__":
    main()
