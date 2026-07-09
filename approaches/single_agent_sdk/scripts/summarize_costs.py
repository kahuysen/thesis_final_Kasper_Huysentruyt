"""Summarize per-image and per-system token usage and wall time across runs.

Sources, in priority order:
  1. <run_dir>/<stem>.meta.json sidecars (written by benchmark_*.py)
  2. /tmp/*_run.log files (parsed for runs that finished pre-sidecar)

Usage:
    .venv/bin/python3 scripts/summarize_costs.py
    .venv/bin/python3 scripts/summarize_costs.py --runs benchmark_runs/run_grok43
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Map run-dir name → /tmp log file (best-effort backfill for older runs)
LOG_HINTS = {
    "run01_opus4.7":             None,                          # very old; logs not preserved
    "run_gpt54":                 None,                          # ditto
    "run_gemini31propreview":    "/tmp/gemini31pro_run.log",
    "run_gemini31propreview-retry": "/tmp/gemini31pro_retry.log",
    "run_gemini3flash":          "/tmp/gemini3flash_run.log",
    "run_grok43":                "/tmp/grok43_run.log",
}

# Log lines look like:
#   [3/16] OK  CEJ_2016.pdf_page001_picture_02_s0.74.png (37.3s, 9 tools, in=10006, out=1574)
LOG_LINE_RX = re.compile(
    r"\[\d+/\d+\]\s+OK\s+(?P<image>\S+)\s+\((?P<elapsed>[0-9.]+)s,\s+(?P<tools>\d+)\s+tools?,"
    r"\s+in=(?P<in>\d+),\s+out=(?P<out>\d+)\)"
)


def from_sidecars(run_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(run_dir.glob("*.meta.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    return rows


def from_log(log_path: Path) -> list[dict]:
    rows = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text(errors="ignore").splitlines():
        m = LOG_LINE_RX.search(line)
        if not m:
            continue
        rows.append(
            {
                "stem": Path(m["image"]).stem,
                "elapsed_s":     float(m["elapsed"]),
                "tool_calls":    int(m["tools"]),
                "input_tokens":  int(m["in"]),
                "output_tokens": int(m["out"]),
                "steps":         None,
                "model":         None,
                "backend":       None,
            }
        )
    return rows


def summarize_run(run_dir: Path) -> tuple[list[dict], dict, str]:
    rows = from_sidecars(run_dir)
    source = "sidecars"
    if not rows:
        log_path = LOG_HINTS.get(run_dir.name)
        if log_path is not None:
            rows = from_log(Path(log_path))
            source = f"parsed log: {log_path}"
    if not rows:
        return [], {"n": 0}, "no data"

    n           = len(rows)
    in_tot      = sum(r.get("input_tokens")  or 0 for r in rows)
    out_tot     = sum(r.get("output_tokens") or 0 for r in rows)
    elapsed_tot = sum(r.get("elapsed_s")     or 0 for r in rows)
    tool_tot    = sum(r.get("tool_calls")    or 0 for r in rows)
    return rows, {
        "n":              n,
        "elapsed_s_total":  round(elapsed_tot, 1),
        "elapsed_s_mean":   round(elapsed_tot / n, 1),
        "input_tokens_total":   in_tot,
        "input_tokens_mean":    int(in_tot / n) if n else 0,
        "output_tokens_total":  out_tot,
        "output_tokens_mean":   int(out_tot / n) if n else 0,
        "tool_calls_total":  tool_tot,
        "tool_calls_mean":   round(tool_tot / n, 1) if n else 0,
    }, source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Specific run dirs to summarize. If omitted, scans benchmark_runs/.")
    args = ap.parse_args()

    if args.runs:
        run_dirs = [Path(p) for p in args.runs]
    else:
        run_dirs = sorted(p for p in (ROOT / "benchmark_runs").iterdir() if p.is_dir())

    print(f"\n{'run':<32}  {'n':>3}  {'wall (s)':>10}  {'in tok':>10}  {'out tok':>10}  "
          f"{'tools':>6}  source")
    print("-" * 110)
    summaries = []
    for d in run_dirs:
        if not d.exists():
            print(f"{d.name:<32}  (missing)")
            continue
        rows, agg, source = summarize_run(d)
        if agg["n"] == 0:
            print(f"{d.name:<32}  {0:>3}  {'-':>10}  {'-':>10}  {'-':>10}  {'-':>6}  {source}")
            continue
        print(
            f"{d.name:<32}  {agg['n']:>3}  "
            f"{agg['elapsed_s_total']:>10.1f}  "
            f"{agg['input_tokens_total']:>10}  "
            f"{agg['output_tokens_total']:>10}  "
            f"{agg['tool_calls_total']:>6}  {source}"
        )
        summaries.append((d.name, agg))

    # Per-image means table
    print(f"\n{'run':<32}  {'mean wall (s)':>14}  {'mean in':>10}  {'mean out':>10}  {'mean tools':>11}")
    print("-" * 90)
    for name, a in summaries:
        print(
            f"{name:<32}  {a['elapsed_s_mean']:>14.1f}  "
            f"{a['input_tokens_mean']:>10}  {a['output_tokens_mean']:>10}  "
            f"{a['tool_calls_mean']:>11.1f}"
        )

    # JSON dump
    out = ROOT / "benchmark_runs" / "_costs_summary.json"
    out.write_text(json.dumps({n: a for n, a in summaries}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
