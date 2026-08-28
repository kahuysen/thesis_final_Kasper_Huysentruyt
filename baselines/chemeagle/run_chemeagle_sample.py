"""Crash-isolated batch driver for ChemEagle over a list of benchmark images.

Each image runs in its own subprocess (`_run_one_image_openrouter.py`) with a
wall-clock timeout, so segfaults/hangs in the vendored stack cost one image,
not the batch. Skips images whose output JSON already exists (resumable).

Usage:
    .venv-chemeagle/bin/python3 run_chemeagle_sample.py \
        --list sample50.txt --out runs/sample50 [--timeout 1200]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parent.parent / "data" / "benchmark_full" / "images"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="text file, one image filename per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in open(HERE / args.list) if l.strip()]
    py = HERE / ".venv-chemeagle" / "bin" / "python3"

    results = []
    for i, name in enumerate(names, 1):
        stem = Path(name).stem
        out_json = out / f"{stem}.json"
        if out_json.exists():
            print(f"[{i}/{len(names)}] SKIP {name}", flush=True)
            continue
        t0 = time.time()
        try:
            proc = subprocess.run(
                [str(py), str(HERE / "_run_one_image_openrouter.py"),
                 str(IMAGES / name), str(out_json)],
                capture_output=True, text=True, timeout=args.timeout, cwd=HERE)
            status = "ok" if proc.returncode == 0 else "err"
            tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        except subprocess.TimeoutExpired:
            status, tail = "timeout", [f"exceeded {args.timeout}s"]
        dt = time.time() - t0
        results.append({"image": name, "status": status, "elapsed_s": round(dt, 1)})
        print(f"[{i}/{len(names)}] {status.upper()} {name} ({dt:.0f}s) {tail[0][:120]}",
              flush=True)

    (out / "_batch_summary.json").write_text(json.dumps(results, indent=1))
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone: {n_ok} ok / {len(results)} attempted "
          f"(+ skipped already-done)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
