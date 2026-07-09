#!/usr/bin/env python3
"""VisualHeist + PDF rasterization driver — runs inside .venv-chemeagle.

Adapted from the ChemEagle baseline's `pdfmodel/methods.py`. We rasterize
the PDF with pypdfium2 (no Poppler dep), then run a Florence-2 fine-tune
(shixuanleong/visualheist-large / -base) to detect figure/table bounding
boxes per page, and crop each detection to its own PNG.

Protocol:

    stdin (one JSON object):
        {
          "pdf_path":   "...",
          "out_dir":    "...",         # where crops go
          "model_size": "large"|"base" # default "large"
          "render_scale": 2.0          # pypdfium2 render scale (default 2.0)
        }

    stdout (one JSON object per line, line-buffered):
        {"event":"start", "total_pages": N}
        {"event":"loading_model", "model_size": "large"}
        {"event":"model_ready"}
        {"event":"page_done", "page": int, "figures": [{path, bbox, page, index}]}
        {"event":"error", "message": str, "trace": str?}      # last event on failure
        {"event":"done", "figures": [...all...]}              # last event on success
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def emit(d: dict) -> None:
    sys.stdout.write(json.dumps(d) + "\n")
    sys.stdout.flush()


def main() -> int:
    try:
        cfg = json.loads(sys.stdin.read())
    except Exception as exc:
        emit({"event": "error", "message": f"bad stdin: {exc}"})
        return 1

    pdf_path = cfg.get("pdf_path")
    out_dir  = Path(cfg.get("out_dir") or ".")
    model_size = (cfg.get("model_size") or "large").lower()
    render_scale = float(cfg.get("render_scale", 2.0))

    if not pdf_path or not os.path.exists(pdf_path):
        emit({"event": "error", "message": f"pdf_path not found: {pdf_path!r}"})
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        n_pages = len(pdf)
        emit({"event": "start", "total_pages": n_pages})

        emit({"event": "loading_model", "model_size": model_size})
        from transformers import AutoModelForCausalLM, AutoProcessor
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        model_id = (
            "shixuanleong/visualheist-large"
            if model_size == "large"
            else "shixuanleong/visualheist-base"
        )
        safetensors_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        state_dict = load_file(safetensors_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, state_dict=state_dict, trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        emit({"event": "model_ready"})

        all_figures: list[dict] = []
        for i in range(n_pages):
            page = pdf[i]
            pil = page.render(scale=render_scale).to_pil().convert("RGB")
            prompt = "<OD>"
            inputs = processor(text=prompt, images=pil, return_tensors="pt")
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            annotation = processor.post_process_generation(
                generated_text, task="<OD>", image_size=(pil.width, pil.height)
            )["<OD>"]

            page_figures: list[dict] = []
            for k, bbox in enumerate(annotation["bboxes"]):
                x1, y1, x2, y2 = bbox
                # Clamp + sanity check
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(pil.width, int(x2)), min(pil.height, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = pil.crop((x1, y1, x2, y2))
                fname = f"page_{i+1:03d}_fig_{k+1:02d}.png"
                fpath = out_dir / fname
                crop.save(fpath)
                page_figures.append({
                    "path":  str(fpath),
                    "bbox":  [x1, y1, x2, y2],
                    "page":  i + 1,
                    "index": k + 1,
                })

            emit({"event": "page_done", "page": i + 1, "figures": page_figures})
            all_figures.extend(page_figures)

        emit({"event": "done", "figures": all_figures})
        return 0

    except Exception as exc:
        emit({
            "event": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(),
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
