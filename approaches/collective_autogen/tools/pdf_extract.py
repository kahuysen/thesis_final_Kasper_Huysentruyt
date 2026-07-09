"""PDF → cropped figure/table images via the Florence-2 fine-tune that
ChemEagle calls "VisualHeist" (`shixuanleong/visualheist-large` |
`-base`).

Wraps the lower-level pieces in `pdfmodel/methods.py`. Unlike the original
`_pdf_to_figures_and_tables`, this returns structured metadata (page,
bbox, label, crop path) so a downstream orchestrator can subprocess
main.py against each crop.

The Florence-2 weights download from HuggingFace on first call (~770 MB
for large, ~270 MB for base) into the standard HF cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .trace import trace


@dataclass
class PdfFigure:
    page: int                # 1-indexed page number
    figure_index: int        # 1-indexed within the page
    label: str               # 'figure' or 'table' (or other Florence-2 label)
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 in page-image pixels
    image_path: Path         # absolute path to the cropped PNG
    page_size: tuple[int, int]        # (width, height) of the source page image


@lru_cache(maxsize=2)
def _load_florence(model_size: str = "large"):
    """Load Florence-2 + processor. Cached so multiple calls reuse weights.

    Loads on whichever device transformers picks (defaults to CPU; you can
    move to MPS/CUDA after construction). Florence-2 inference is fast
    enough on CPU for small papers (a few seconds per page) so we don't
    bother with device placement here.
    """
    from pdfmodel.methods import _create_model
    if model_size not in ("base", "large"):
        raise ValueError(f"model_size must be 'base' or 'large', got {model_size!r}")
    model_id = (
        "shixuanleong/visualheist-large" if model_size == "large"
        else "shixuanleong/visualheist-base"
    )
    return _create_model(model_id, model_size)


def _parse_pages(pages: str | None, total: int) -> list[int]:
    """Parse "3-5,7" / "3" / None into a sorted list of 1-indexed pages.

    None or empty → every page.
    """
    if not pages:
        return list(range(1, total + 1))
    out: set[int] = set()
    for chunk in pages.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            lo, hi = int(a), int(b)
            for n in range(min(lo, hi), max(lo, hi) + 1):
                if 1 <= n <= total:
                    out.add(n)
        else:
            n = int(chunk)
            if 1 <= n <= total:
                out.add(n)
    return sorted(out)


@trace()
def extract_figures_from_pdf(
    pdf_path: str,
    output_dir: str,
    pages: str | None = None,
    model_size: str = "large",
    dpi: int = 200,
) -> dict:
    """Detect figures and tables in a PDF and save each as a cropped PNG.

    Args:
        pdf_path: path to the source PDF.
        output_dir: directory to write cropped PNGs into. Created if missing.
        pages: comma/range page filter, e.g. "3-5,7". None → all pages.
        model_size: "large" (default) or "base".
        dpi: rasterisation DPI for pdf2image; 200 matches ChemEagle.

    Returns:
        dict with keys:
            figures        — list of PdfFigure-shaped dicts (see _to_dict below).
            pages_processed — list of 1-indexed page numbers actually run.
            page_count     — total pages in the source PDF.
            error          — None on success, a string on failure.

    Crop filenames: <pdf_stem>_p<page>_<label>_<idx>.png
    """
    from pdf2image import convert_from_path

    pdf = Path(pdf_path).resolve()
    if not pdf.exists():
        return {"figures": [], "pages_processed": [], "page_count": 0,
                "error": f"pdf not found: {pdf}"}

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_pages = convert_from_path(str(pdf), dpi=dpi)
    except Exception as e:
        return {"figures": [], "pages_processed": [], "page_count": 0,
                "error": f"pdf2image failed: {type(e).__name__}: {e}"}

    page_indices = _parse_pages(pages, total=len(all_pages))
    if not page_indices:
        return {"figures": [], "pages_processed": [], "page_count": len(all_pages),
                "error": f"no pages selected from filter {pages!r} (pdf has {len(all_pages)} pages)"}

    try:
        model, processor = _load_florence(model_size)
    except Exception as e:
        return {"figures": [], "pages_processed": [], "page_count": len(all_pages),
                "error": f"Florence-2 load failed: {type(e).__name__}: {e}"}

    from pdfmodel.methods import _tf_id_detection

    figures: list[dict] = []
    stem = pdf.stem
    for page_no in page_indices:
        page_img = all_pages[page_no - 1]
        page_size = page_img.size
        try:
            ann = _tf_id_detection(page_img, model, processor)
        except Exception as e:
            figures.append({
                "page": page_no, "figure_index": 0, "label": "_error",
                "bbox": [0, 0, 0, 0], "image_path": "",
                "page_size": list(page_size),
                "error": f"detection failed on page {page_no}: {type(e).__name__}: {e}",
            })
            continue

        bboxes = ann.get("bboxes", []) or []
        labels = ann.get("labels", []) or [""] * len(bboxes)
        for i, (bbox, label) in enumerate(zip(bboxes, labels), start=1):
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(page_size[0], x2); y2 = min(page_size[1], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            label_str = (label or "figure").replace("/", "_").replace(" ", "_")
            crop_name = f"{stem}_p{page_no:03d}_{label_str}_{i:02d}.png"
            crop_path = out_dir / crop_name
            page_img.crop((x1, y1, x2, y2)).save(crop_path, "PNG")
            figures.append({
                "page": page_no,
                "figure_index": i,
                "label": label_str,
                "bbox": [x1, y1, x2, y2],
                "image_path": str(crop_path),
                "page_size": list(page_size),
            })

    return {
        "figures": figures,
        "pages_processed": page_indices,
        "page_count": len(all_pages),
        "error": None,
    }
