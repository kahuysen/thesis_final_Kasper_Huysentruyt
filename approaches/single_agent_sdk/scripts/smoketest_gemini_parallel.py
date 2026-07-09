"""Smoke test: confirm the parallel Gemini extractor batches calls per turn.

Runs `extract_figure_stream_gemini_parallel` on a single image and reports
total tool calls, round-trips, calls-per-round-trip, latency and tokens.
A calls-per-round-trip ratio > 1 confirms parallel function calling is
happening (Gemini 3 emitted multiple function_call parts in one response).

Usage:
    .venv/bin/python3 scripts/smoketest_gemini_parallel.py
    .venv/bin/python3 scripts/smoketest_gemini_parallel.py --image <path>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from google import genai

from pipeline.extractor import SYSTEM_PROMPT
from pipeline.extractor_gemini_parallel import extract_figure_stream_gemini_parallel


DEFAULT_IMAGE = ROOT / "corpus/Benchmark_kasper_GT3_Maarten/ACScat_2020.pdf_page002_table_01_s0.88.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--model", default="gemini-3-flash-preview")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        return 1

    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    print(f"Image: {args.image.name}")
    print(f"Model: {args.model}\n")

    t0 = time.time()
    tool_calls = 0
    metadata = None
    extraction = None
    error = None
    for ev in extract_figure_stream_gemini_parallel(
        args.image, model=args.model, client=client, system_prompt=SYSTEM_PROMPT
    ):
        kind = ev.get("event")
        if kind == "step_done":
            tool_calls += 1
        elif kind == "complete":
            metadata = ev["metadata"]
            extraction = ev.get("extraction")
        elif kind == "error":
            error = ev["message"]

    elapsed = time.time() - t0
    steps = metadata["steps"] if metadata else None
    n_rxn = len(extraction.get("reactions", [])) if extraction else 0
    ratio = tool_calls / steps if steps else 0

    print("=" * 60)
    print(f"  tool_calls           : {tool_calls}")
    print(f"  round_trips          : {steps}")
    print(f"  calls per round-trip : {ratio:.2f}")
    print(f"  elapsed_s            : {elapsed:.1f}")
    if metadata:
        print(f"  input_tokens         : {metadata.get('input_tokens')}")
        print(f"  output_tokens        : {metadata.get('output_tokens')}")
    print(f"  reactions submitted  : {n_rxn}")
    print(f"  error                : {error}")
    print("=" * 60)
    if ratio > 1.5:
        print(f"\nPARALLEL: {ratio:.2f} calls per round-trip — Gemini is batching.")
    elif steps:
        print(f"\nSEQUENTIAL: {ratio:.2f} calls per round-trip — model emitted "
              "one call at a time on this image.")
    return 0 if error is None else 1


if __name__ == "__main__":
    sys.exit(main())
