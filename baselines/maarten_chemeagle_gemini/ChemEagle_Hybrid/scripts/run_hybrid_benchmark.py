"""Run ChemEagle_Hybrid (molnextr+rxnim+Gemini) over the article benchmark.

Mirrors scripts/run_gemini_benchmark.py with three differences:
- Instantiates ChemIEToolkit once at startup (lazy-loads molnextr/rxnim, ~30s).
- Default --parallel 1 (single A6000; molnextr/rxnim contend for GPU).
- Default --output / --trace-dir don't collide with the v3 pure-Gemini run.

Output JSON shape (per scripts/benchmark_eval.py):
    { "0": {"reactions": [...]}, "1": {"reactions": [...]}, ... }
"""

import argparse
import getpass
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)


def _atomic_write_json(path: str, obj) -> None:
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_existing(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[WARN] existing --output unreadable ({e}); starting fresh", file=sys.stderr)
    return {}


def _fmt_t(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _resolve_api_key(args) -> str:
    if args.api_key == "-" or args.api_key_stdin:
        key = getpass.getpass("Gemini API key: ").strip()
        if not key:
            print("[ERR] no API key entered", file=sys.stderr)
            sys.exit(2)
        return key
    if args.api_key:
        return args.api_key.strip()
    env = os.getenv("GEMINI_API_KEY", "").strip()
    if env:
        return env
    print("[ERR] provide Gemini API key via --api-key, --api-key-stdin, or GEMINI_API_KEY", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt",
        default="input/Benchmark_chemeagle/Benchmark_data/article/GT3.json",
    )
    parser.add_argument(
        "--images-dir",
        default="input/Benchmark_chemeagle/Benchmark_data/article",
    )
    parser.add_argument(
        "--output",
        default="output/benchmark_article_hybrid.json",
    )
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument(
        "--api-key",
        default=None,
        help='Gemini API key. Use "-" to be prompted (hidden input).',
    )
    parser.add_argument("--api-key-stdin", action="store_true", help="Prompt for the key (hidden input).")
    parser.add_argument("--trace-dir", default=None, help="Default: output/traces/<run_id>_hybrid/")
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Max concurrent images (default 1; >1 risks GPU OOM on molnextr/rxnim).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable explicit context caching (sends prompt inline).")
    parser.add_argument("--cache-ttl-seconds", type=int, default=3600)
    args = parser.parse_args()

    api_key = _resolve_api_key(args)

    os.environ["CHEMEAGLE_BACKEND"] = "gemini"
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["CHEMEAGLE_MODEL"] = args.model

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = args.trace_dir or os.path.join("output", "traces", f"{run_id}_hybrid")

    # Imports after env is set.
    from gemini_hybrid import ChemEagle_Hybrid, create_hybrid_cache  # noqa: E402
    from gemini_extractor import create_variant_cache                # noqa: E402
    from utils.llm_client import (                                   # noqa: E402
        clear_image_context, get_client, get_global_usage,
        get_image_usage, set_image_context, set_trace_dir,
    )
    set_trace_dir(trace_dir)

    with open(args.gt, "r", encoding="utf-8") as f:
        gt = json.load(f)
    if not isinstance(gt, list):
        print(f"[ERR] --gt must be a list, got {type(gt).__name__}", file=sys.stderr)
        return 2
    n_total = len(gt)
    end = args.end if args.end is not None else n_total
    indices = list(range(args.start, min(end, n_total)))
    if args.limit is not None:
        indices = indices[: args.limit]

    results = _load_existing(args.output) if args.resume else {}
    if args.resume:
        print(f"[INFO] resuming: {len(results)} indices already in {args.output}")

    print(f"[RUN] model={args.model}  trace_dir={trace_dir}  images={len(indices)}  parallel={args.parallel}")

    # --- toolkit init (lazy-loads molnextr, rxnim, coref) ----------------
    print("[INIT] instantiating ChemIEToolkit (first molnextr/rxnim call will download/load weights)...")
    t_init = time.perf_counter()
    import torch  # noqa: E402
    from chemietoolkit import ChemIEToolkit  # noqa: E402
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    toolkit = ChemIEToolkit(device=device)
    # Touch the lazy-loaded properties so the load cost is counted at init,
    # not on the first image (which would skew the per-image timing).
    _ = toolkit.molnextr; _ = toolkit.rxnim; _ = toolkit.coref
    print(f"[INIT] toolkit ready (device={device}) in {_fmt_t(time.perf_counter() - t_init)}")

    # --- cache setup -----------------------------------------------------
    client = get_client(api_key=api_key)
    extract_cache_name = None
    variant_cache_name = None
    if not args.no_cache and hasattr(client, "create_cache"):
        extract_cache_name = create_hybrid_cache(
            client, model=args.model, ttl_seconds=args.cache_ttl_seconds,
        )
        variant_cache_name = create_variant_cache(
            client, model=args.model, ttl_seconds=args.cache_ttl_seconds,
        )
        print(
            f"[CACHE] hybrid={extract_cache_name or 'FAILED'}  "
            f"variant={variant_cache_name or 'FAILED'}  "
            f"ttl={args.cache_ttl_seconds}s"
        )
    else:
        print("[CACHE] disabled (no-cache flag or unsupported client).")

    write_lock = threading.Lock()
    t0 = time.perf_counter()
    n_fail = 0
    n_done = 0

    def _process(idx: int):
        key = str(idx)
        entry = gt[idx]
        file_name = entry.get("file_name") or f"image_{idx}"
        if args.resume and key in results:
            return ("skip", idx, file_name, None, 0.0, None)
        image_path = os.path.join(args.images_dir, file_name)
        set_image_context(key)
        tic = time.perf_counter()
        pred = None
        raised = False
        err_msg = None
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            pred = ChemEagle_Hybrid(
                image_path,
                toolkit=toolkit,
                cache_name=extract_cache_name,
                variant_cache_name=variant_cache_name,
                model=args.model,
                api_key=api_key,
            )
            if not isinstance(pred, dict):
                raise TypeError(f"ChemEagle_Hybrid returned {type(pred).__name__}")
            pred.setdefault("reactions", [])
        except Exception as e:
            raised = True
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {idx:>3} {file_name}: {err_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            pred = {"reactions": [], "_error": err_msg}
        dt = time.perf_counter() - tic
        tokens = get_image_usage()
        clear_image_context()
        with write_lock:
            results[key] = pred
            _atomic_write_json(args.output, results)
        return ("fail" if raised else "ok", idx, file_name, pred, dt, tokens)

    def _print_row(res, i, total):
        status, idx, file_name, pred, dt, tokens = res
        if status == "skip":
            print(f"[SKIP] {idx:>3} {file_name}  (already done)")
            return
        n_rx = len(pred.get("reactions") or []) if pred else 0
        s2 = "+s2" if (pred or {}).get("_stage2_applied") else "   "
        elapsed = time.perf_counter() - t0
        avg = elapsed / max(i, 1)
        remaining = (total - i) * avg
        tok = tokens or {}
        print(
            f"[{status.upper():>4}{s2}] {idx:>3} {file_name}  "
            f"rxns={n_rx}  {dt:5.1f}s  "
            f"tok p/c/t={tok.get('prompt_tokens',0)}/"
            f"{tok.get('completion_tokens',0)}/{tok.get('total_tokens',0)}  "
            f"calls={tok.get('calls',0)}  "
            f"elapsed={_fmt_t(elapsed)}  eta={_fmt_t(remaining)}"
        )

    try:
        if args.parallel <= 1:
            for i, idx in enumerate(indices, start=1):
                res = _process(idx)
                _print_row(res, i, len(indices))
                if res[0] == "fail":
                    n_fail += 1
                if res[0] != "skip":
                    n_done += 1
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futures = {ex.submit(_process, idx): idx for idx in indices}
                for i, fut in enumerate(as_completed(futures), start=1):
                    res = fut.result()
                    _print_row(res, i, len(futures))
                    if res[0] == "fail":
                        n_fail += 1
                    if res[0] != "skip":
                        n_done += 1
    finally:
        # Always delete caches.
        if extract_cache_name and hasattr(client, "delete_cache"):
            try:
                client.delete_cache(extract_cache_name)
                print(f"[CACHE] deleted {extract_cache_name}")
            except Exception as e:
                print(f"[CACHE] delete failed for {extract_cache_name}: {e}", file=sys.stderr)
        if variant_cache_name and hasattr(client, "delete_cache"):
            try:
                client.delete_cache(variant_cache_name)
                print(f"[CACHE] deleted {variant_cache_name}")
            except Exception as e:
                print(f"[CACHE] delete failed for {variant_cache_name}: {e}", file=sys.stderr)

    total = time.perf_counter() - t0
    gu = get_global_usage()
    print(
        f"\n[DONE] {n_done} processed ({n_fail} failed) in {_fmt_t(total)}  ->  {args.output}\n"
        f"[USAGE] calls={gu['calls']}  "
        f"prompt_tokens={gu['prompt_tokens']}  "
        f"completion_tokens={gu['completion_tokens']}  "
        f"total_tokens={gu['total_tokens']}\n"
        f"[TRACE] per-image JSONL logs: {trace_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
