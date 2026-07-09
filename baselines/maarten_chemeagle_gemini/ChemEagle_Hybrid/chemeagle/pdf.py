"""End-to-end: PDF -> per-figure reaction JSON via gemini_hybrid.

Pipeline:
  1. Read PDF text layer per page (PyMuPDF / fitz).
  2. Crop figures+tables out of the PDF (pdf_extraction.run_pdf -> VisualHeist).
  3. Build one ChemIEToolkit (lazy-loads molnextr/rxnim/coref via HF Hub).
  4. Create one Gemini context cache for the hybrid orchestrate prompt.
  5. Loop ChemEagle_Hybrid over each cropped PNG with the cache attached.
  6. Aggregate into one dict; clean up cache and (optionally) PNGs.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional


def parse_pdf(
    pdf_path: str | os.PathLike,
    *,
    out_dir: Optional[str | os.PathLike] = None,
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    keep_figures: bool = False,
) -> dict[str, Any]:
    """Parse a chemical PDF end-to-end with the hybrid Gemini backend.

    Parameters
    ----------
    pdf_path : path to a digital-native chemistry PDF (text layer required for
        body-text extraction; scanned PDFs are not supported).
    out_dir : where cropped figure PNGs are written. A tempdir is used if None.
    model : Gemini model name. Default matches gemini_hybrid.
    api_key : Gemini API key. Falls back to GEMINI_API_KEY env var.
    keep_figures : if True and out_dir was None, the tempdir is preserved.

    Returns
    -------
    dict with shape:
      {
        "pdf_path": str,
        "model": str,
        "pages": [{"page": int, "text": str}, ...],
        "figures": [{"png": str, "result": {...hybrid output...}}, ...],
        "usage": {"prompt_tokens": int, "completion_tokens": int,
                  "total_tokens": int, "calls": int},
      }
    """
    pdf_path = str(pdf_path)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)

    # Gemini backend requires this env var BEFORE get_client is called.
    os.environ["CHEMEAGLE_BACKEND"] = "gemini"
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Pass api_key=... or export GEMINI_API_KEY."
        )

    # Imports deferred until after env vars are set, since some legacy modules
    # read env at import time.
    import fitz  # PyMuPDF
    from chemietoolkit import ChemIEToolkit
    from gemini_hybrid import ChemEagle_Hybrid, create_hybrid_cache
    from pdf_extraction import run_pdf
    from utils.llm_client import get_client, get_global_usage

    # Stage 1: text layer per page.
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append({"page": i, "text": page.get_text("text") or ""})

    # Stage 2: crop figures+tables out of the PDF.
    using_tmp = out_dir is None
    if using_tmp:
        out_dir = tempfile.mkdtemp(prefix="chemeagle_figs_")
    else:
        out_dir = str(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    try:
        run_pdf(pdf_dir=pdf_path, image_dir=out_dir)

        png_files = sorted(
            f for f in os.listdir(out_dir) if f.lower().endswith(".png")
        )

        # Stage 3: shared toolkit (lazy-loads molnextr/rxnim/coref via HF Hub).
        toolkit = ChemIEToolkit()

        # Stage 4: prompt cache for the hybrid orchestrate system prompt.
        client = get_client()
        cache_name = create_hybrid_cache(client, model=model)

        # Stage 5: per-figure extraction.
        figures: list[dict[str, Any]] = []
        try:
            for fname in png_files:
                png_path = os.path.join(out_dir, fname)
                result = ChemEagle_Hybrid(
                    png_path,
                    toolkit=toolkit,
                    cache_name=cache_name,
                    model=model,
                )
                figures.append({"png": fname, "result": result})
        finally:
            # Always release the cache; it ticks against the Gemini caches quota.
            try:
                client.delete_cache(cache_name)
            except Exception:
                pass
    finally:
        if using_tmp and not keep_figures:
            shutil.rmtree(out_dir, ignore_errors=True)

    return {
        "pdf_path": pdf_path,
        "model": model,
        "pages": pages,
        "figures": figures,
        "usage": get_global_usage(),
    }


def write_result(result: dict, out_path: str | os.PathLike) -> None:
    """Write the parse_pdf result to a JSON file (utf-8, indented)."""
    Path(out_path).write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
