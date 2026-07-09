"""PDF entrypoint — extract every figure/table from a paper, then run main.py
once per crop.

Mirrors the pattern in eval/run_benchmark_suite.py: a thin orchestrator that
subprocesses main.py per image, with one extra step up front (Florence-2
figure detection) that turns a PDF into a working set of cropped PNGs.

Usage:
    python main_pdf.py paper.pdf [--pages 3-5,7] [--model-size large|base]
                                 [--max-parallel N] [--keep-crops]
                                 [--filter-labels figure,table]
                                 [-- + any main.py flag, e.g. --vision-model gpt-5.4 --use-molnextr]

Output layout:
    runs/pdf_<ts>_<pdf_stem>/
      crops/<pdf_stem>_p001_figure_01.png ...
      figures.json                       # detection metadata for every crop
      paper_result.json                  # aggregated per-figure run summary
      <crop_stem>/                       # one main.py run dir per figure
        result.json
        conversation.txt
        ...
"""
from __future__ import annotations

# --- _lzma shim — must come BEFORE any transformers / torch / torchvision import.
# The user's pyenv Python 3.12 was built without xz-utils and lacks _lzma.
# transformers 4.47 imports lzma during processing_auto bootstrap.
import sys as _sys
import types as _types

if "_lzma" not in _sys.modules:
    _m = _types.ModuleType("_lzma")
    for _attr in (
        "FORMAT_XZ", "FORMAT_ALONE", "FORMAT_RAW", "FORMAT_AUTO",
        "CHECK_NONE", "CHECK_CRC32", "CHECK_CRC64", "CHECK_SHA256",
        "CHECK_ID_MAX", "CHECK_UNKNOWN",
        "FILTER_LZMA1", "FILTER_LZMA2", "FILTER_DELTA",
        "FILTER_X86", "FILTER_POWERPC", "FILTER_IA64",
        "FILTER_ARM", "FILTER_ARMTHUMB", "FILTER_SPARC",
        "MF_HC3", "MF_HC4", "MF_BT2", "MF_BT3", "MF_BT4",
        "MODE_FAST", "MODE_NORMAL",
        "PRESET_DEFAULT", "PRESET_EXTREME",
    ):
        setattr(_m, _attr, 0)

    class _Dummy:
        def __init__(self, *a, **k): pass
        def decompress(self, *a, **k): return b""
        def compress(self, *a, **k): return b""
        def flush(self, *a, **k): return b""

    _m.LZMADecompressor = _Dummy
    _m.LZMACompressor = _Dummy

    class _Err(Exception): pass
    _m.LZMAError = _Err
    _m.is_check_supported = lambda x: False
    _m._encode_filter_properties = lambda x: b""
    _m._decode_filter_properties = lambda x, y: {}
    _sys.modules["_lzma"] = _m
# ------------------------------------------------------------------------------

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from tools.pdf_extract import extract_figures_from_pdf  # noqa: E402


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split CLI args at the literal `--` separator.

    Everything before `--` is parsed by main_pdf's argparser; everything
    after is forwarded verbatim to each per-figure main.py subprocess.
    """
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _run_one_figure(
    crop_path: Path,
    run_dir: Path,
    main_py: Path,
    forwarded: list[str],
    venv_python: Path,
    timeout_s: int,
) -> dict:
    """Subprocess one main.py invocation against a cropped figure.

    The per-figure run dir is created BEFORE the subprocess starts so we
    can place the log file there even on fast crashes. main.py creates
    its own dated subdir under runs/, so this dir collects metadata only.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"
    cmd = [str(venv_python), "-u", str(main_py), str(crop_path), *forwarded]

    t0 = time.time()
    timed_out = False
    err = None
    try:
        with log_path.open("w") as f:
            proc = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                cwd=str(REPO), timeout=timeout_s,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -2
        timed_out = True
        err = f"timeout after {timeout_s}s"
        # Best-effort kill of orphan main.py descendants.
        try:
            subprocess.run(["pkill", "-KILL", "-f", f"main.py {crop_path}"],
                           capture_output=True)
        except Exception:
            pass
    except Exception as exc:
        rc = -1
        err = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t0

    main_run_dir = _latest_run_for(crop_path.stem, since=t0 - 5)
    result_json = None
    if main_run_dir is not None:
        rj = main_run_dir / "result.json"
        if rj.exists():
            try:
                result_json = json.loads(rj.read_text())
            except Exception:
                result_json = None

    return {
        "crop": crop_path.name,
        "rc": rc,
        "elapsed_s": round(elapsed, 1),
        "timed_out": timed_out,
        "error": err,
        "log": str(log_path),
        "main_run_dir": str(main_run_dir) if main_run_dir else None,
        "result": result_json,
    }


def _latest_run_for(crop_stem: str, since: float) -> Path | None:
    """Find the runs/<ts>_<team> dir whose result.json lists crop_stem.

    main.py writes runs to <repo>/runs/<YYYYMMDD_HHMMSS>_<team>/. We pick
    the most recent dir created after `since` whose result.json's
    file_name matches.
    """
    runs_root = REPO / "runs"
    if not runs_root.exists():
        return None
    for d in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        if d.stat().st_mtime < since:
            break
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
            if Path(data.get("file_name", "")).stem == crop_stem:
                return d
        except Exception:
            continue
    return None


def main() -> int:
    own_argv, forwarded = _split_argv(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Run the multi-agent pipeline on every figure/table in a PDF.",
        epilog="Pass any main.py flag after `--`, e.g. "
               "`python main_pdf.py paper.pdf -- --use-molnextr --vision-model gpt-5.4`",
    )
    parser.add_argument("pdf_path", help="Path to the source PDF.")
    parser.add_argument("--pages", default=None,
                        help="Page filter: '3-5,7' or '4'. Default: all pages.")
    parser.add_argument("--model-size", choices=("base", "large"), default="large",
                        help="Florence-2 size (default: large).")
    parser.add_argument("--max-parallel", type=int, default=3,
                        help="Concurrent main.py subprocesses (default: 3). "
                             "Each one makes its own LLM calls and may load MolNexTR — "
                             "high values can hit Azure rate limits or exhaust GPU memory.")
    parser.add_argument("--per-figure-timeout", type=int, default=900,
                        help="Hard wall-clock timeout per figure in seconds (default 900).")
    parser.add_argument("--keep-crops", action="store_true",
                        help="Keep the crops/ directory after the run (default: keep).")
    parser.add_argument("--filter-labels", default=None,
                        help="Comma-separated Florence-2 labels to keep (e.g. 'figure,table'). "
                             "Default: keep everything.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect figures and write figures.json, then exit without "
                             "running main.py.")
    args = parser.parse_args(own_argv)

    pdf = Path(args.pdf_path).resolve()
    if not pdf.exists():
        print(f"PDF not found: {pdf}", file=sys.stderr)
        return 1

    started_at = datetime.now()
    out_dir = REPO / "runs" / f"pdf_{started_at.strftime('%Y%m%d_%H%M%S')}_{pdf.stem}"
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PDF run {out_dir.name} ===")
    print(f"  source: {pdf}")
    print(f"  pages:  {args.pages or 'all'}")
    print(f"  model:  visualheist-{args.model_size}")
    print(f"  out:    {out_dir.relative_to(REPO)}")
    print()

    print("[1/2] Florence-2 figure detection ...")
    extraction = extract_figures_from_pdf(
        str(pdf), str(crops_dir), pages=args.pages, model_size=args.model_size,
    )
    if extraction.get("error"):
        print(f"  detection failed: {extraction['error']}", file=sys.stderr)
        (out_dir / "figures.json").write_text(
            json.dumps(extraction, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 2

    figures = extraction["figures"]
    if args.filter_labels:
        keep = {s.strip() for s in args.filter_labels.split(",") if s.strip()}
        figures = [f for f in figures if f.get("label") in keep]

    (out_dir / "figures.json").write_text(
        json.dumps({**extraction, "figures": figures}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  pages processed: {extraction['pages_processed']}")
    print(f"  figures detected: {len(figures)}")
    if args.filter_labels:
        print(f"  after label filter ({args.filter_labels}): {len(figures)}")

    if not figures:
        print("  no figures to run; exiting.")
        return 0

    if args.dry_run:
        print("  --dry-run set; skipping main.py runs.")
        return 0

    print()
    print(f"[2/2] Running main.py on {len(figures)} crop(s) "
          f"(parallel={args.max_parallel}) ...")

    main_py = REPO / "main.py"
    venv_python = Path(sys.executable)

    # Pin parallel ≥ 1.
    workers = max(1, int(args.max_parallel))

    per_figure: list[dict] = []
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fig in figures:
            crop_path = Path(fig["image_path"])
            run_dir = out_dir / crop_path.stem
            fut = pool.submit(
                _run_one_figure, crop_path, run_dir,
                main_py, forwarded, venv_python, args.per_figure_timeout,
            )
            futures[fut] = fig

        completed = 0
        for fut in as_completed(futures):
            fig = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"crop": Path(fig["image_path"]).name, "rc": -1,
                       "elapsed_s": 0.0, "timed_out": False,
                       "error": f"{type(exc).__name__}: {exc}",
                       "log": None, "main_run_dir": None, "result": None}
            row.update({
                "page": fig["page"],
                "figure_index": fig["figure_index"],
                "label": fig["label"],
                "bbox": fig["bbox"],
            })
            per_figure.append(row)
            completed += 1
            flag = "TIMEOUT" if row["timed_out"] else f"rc={row['rc']}"
            print(f"  [{completed}/{len(figures)}] {row['crop']:50s}  {flag}  {row['elapsed_s']:5.0f}s")

    paper_summary = {
        "pdf_path": str(pdf),
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now().isoformat(),
        "pages_processed": extraction["pages_processed"],
        "page_count": extraction["page_count"],
        "model_size": args.model_size,
        "max_parallel": workers,
        "forwarded_main_args": forwarded,
        "figures": sorted(per_figure, key=lambda r: (r["page"], r["figure_index"])),
        "summary": {
            "total": len(per_figure),
            "succeeded": sum(1 for r in per_figure if r["rc"] == 0 and r["result"] is not None),
            "failed": sum(1 for r in per_figure if r["rc"] != 0 or r["result"] is None),
            "timed_out": sum(1 for r in per_figure if r["timed_out"]),
        },
    }
    (out_dir / "paper_result.json").write_text(
        json.dumps(paper_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    s = paper_summary["summary"]
    print()
    print(f"=== {out_dir.name}: {s['succeeded']}/{s['total']} ok"
          f"  ({s['failed']} failed, {s['timed_out']} timed out) ===")
    print(f"  paper_result.json: {(out_dir / 'paper_result.json').relative_to(REPO)}")
    return 0 if s["failed"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
