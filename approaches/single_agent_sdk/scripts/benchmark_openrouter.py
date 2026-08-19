"""Run any model accessible via OpenRouter on the benchmark.

Usage:
    .venv/bin/python3 scripts/benchmark_openrouter.py \
        --model anthropic/claude-opus-4-7 \
        --out  benchmark_runs/run_openrouter_opus

Reads OPENROUTER_API_KEY (and optional OPENROUTER_BASE_URL, default
https://openrouter.ai/api/v1) from .env.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from openai import OpenAI
from pydantic import ValidationError

from pipeline.extractor import SYSTEM_PROMPT
from pipeline.extractor_openrouter import extract_figure_stream_openrouter
from pipeline.schema import FigureExtraction
from scripts._meta_helpers import write_meta_sidecar


def run_one(client, image_path: Path, model: str, out_dir: Path,
            *, max_retries: int = 5, base_backoff: float = 30.0) -> dict:
    stem = image_path.stem
    out_json = out_dir / f"{stem}.json"
    if out_json.exists():
        return {"image": image_path.name, "status": "skip", "reason": "exists"}

    t0 = time.time()
    extraction_dict = None
    metadata = None
    error_msg = None
    tool_calls = 0

    for attempt in range(1, max_retries + 2):
        extraction_dict = None
        metadata = None
        error_msg = None
        tool_calls = 0
        try:
            for ev in extract_figure_stream_openrouter(
                image_path,
                model=model,
                client=client,
                system_prompt=SYSTEM_PROMPT,
            ):
                kind = ev.get("event")
                if kind == "step_done":
                    tool_calls += 1
                elif kind == "complete":
                    extraction_dict = ev["extraction"]
                    metadata = ev["metadata"]
                elif kind == "error":
                    error_msg = ev["message"]
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

        if extraction_dict is not None:
            break
        is_429 = error_msg and (
            "429" in error_msg or "rate" in error_msg.lower()
            or "RESOURCE_EXHAUSTED" in error_msg
        )
        if not is_429 or attempt > max_retries:
            break
        wait_s = base_backoff * attempt
        print(f"    ⟳ rate limit — backing off {wait_s:.0f}s (attempt {attempt}/{max_retries})", flush=True)
        time.sleep(wait_s)

    elapsed = time.time() - t0
    if error_msg or extraction_dict is None:
        return {"image": image_path.name, "status": "err",
                "error": error_msg or "no complete event", "elapsed_s": round(elapsed, 1)}

    try:
        fx = FigureExtraction.model_validate(extraction_dict)
    except ValidationError as e:
        return {"image": image_path.name, "status": "err",
                "error": f"schema: {e.errors()[:1]}", "elapsed_s": round(elapsed, 1)}

    out_json.write_text(fx.model_dump_json(indent=2))
    write_meta_sidecar(out_dir, stem, metadata=metadata, elapsed_s=elapsed,
                       tool_calls=tool_calls, backend="openrouter", model=model)
    return {"image": image_path.name, "status": "ok",
            "elapsed_s": round(elapsed, 1), "tool_calls": tool_calls,
            "metadata": metadata}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/Benchmark_kasper_GT3_Maarten")
    ap.add_argument("--out", required=True,
                    help="e.g. benchmark_runs/run_openrouter_<model-tag>")
    ap.add_argument("--model", required=True,
                    help="OpenRouter model id, e.g. anthropic/claude-opus-4-7, "
                         "openai/gpt-5.4, google/gemini-3-pro-preview, openrouter/auto")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set in .env")
        return 2
    base_url = args.base_url or os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )

    client = OpenAI(api_key=api_key, base_url=base_url)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    images = sorted(p for p in corpus.iterdir()
                    if p.is_file() and p.suffix.lower() in exts)
    if not images:
        print(f"No images under {args.corpus}")
        return 1

    print(f"Backend:  openrouter")
    print(f"  endpoint: {base_url}")
    print(f"  model:    {args.model}")
    print(f"Images:   {args.corpus}  ({len(images)} files)")
    print(f"Out:      {out_dir}\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(run_one, client, img, args.model, out_dir): img for img in images}
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            results.append(res)
            if res["status"] == "skip":
                tag, tail = "SKIP", ""
            elif res["status"] == "ok":
                m = res.get("metadata") or {}
                tag = "OK"
                tail = (f" ({res['elapsed_s']}s, {res.get('tool_calls','?')} tools, "
                        f"in={m.get('input_tokens','?')}, out={m.get('output_tokens','?')})")
            else:
                tag = "ERR"
                tail = f" {res.get('error','')[:200]}"
            print(f"[{i}/{len(images)}] {tag}  {res['image']}{tail}", flush=True)
            if args.delay > 0 and res.get("status") != "skip":
                time.sleep(args.delay)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = sum(1 for r in results if r["status"] == "err")
    print(f"\nDone: {n_ok} ok, {n_err} failed, {len(results)} total")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
