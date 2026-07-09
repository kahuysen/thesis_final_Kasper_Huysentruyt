#!/usr/bin/env python3
"""Verify that path expressions inside the rewritten source files resolve.

Runs with system Python (>= 3.8) and stdlib only — no venv needed.

Three groups of checks:
  1. Path-constant resolution: simulate the `COLL = ROOT.parent / ...` style
     expressions in the 5 files we edited and confirm each resolves to an
     existing directory.
  2. Compile-check: every .py file under approaches/ parses with `compile()`.
  3. Module-availability: the modules the rewritten scripts import (e.g.
     `eval.metrics`, `scripts.eval_via_collective`) exist on disk at the
     locations the new path expressions point to.

Run from anywhere:
    python tests/check_paths.py
Exit 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

ROOT = Path(__file__).resolve().parent.parent  # 5.Code_reorg/

passed = 0
failed: list[str] = []


def check(desc: str, cond: bool, detail: str = "") -> None:
    global passed
    if cond:
        passed += 1
        print(f"  {GREEN}✓{RESET} {desc}")
    else:
        failed.append(desc + (f" ({detail})" if detail else ""))
        print(f"  {RED}✗{RESET} {desc}" + (f"  [{detail}]" if detail else ""))


def section(name: str) -> None:
    print(f"\n{BOLD}{name}{RESET}")


# --------------------------------------------------------------------------
# 1. Path-constant resolution
# --------------------------------------------------------------------------
section("Path constants in rewritten scripts resolve to existing dirs")

# Simulate plot_cost_quality.py / plot_aggregate_heatmap.py / append_to_eval_xlsx.py:
#   ROOT = Path(__file__).resolve().parent.parent            # → single_agent_sdk/
#   COLL = ROOT.parent / "collective_autogen"                # → approaches/collective_autogen/
sdk_scripts = ROOT / "approaches" / "single_agent_sdk" / "scripts"
for script_name in (
    "plot_cost_quality.py",
    "plot_aggregate_heatmap.py",
    "append_to_eval_xlsx.py",
):
    script = sdk_scripts / script_name
    if not script.exists():
        check(f"{script_name} present", False, f"not at {script}")
        continue
    script_root = script.resolve().parent.parent  # mirrors ROOT in those files
    coll = script_root.parent / "collective_autogen"
    check(f"{script_name}: ROOT exists ({script_root.name})", script_root.is_dir())
    check(f"{script_name}: COLL resolves to existing dir", coll.is_dir(),
          f"computed {coll}")

# eval_via_collective.py uses a 3-parent walk instead of ROOT.parent:
#   COLL = Path(__file__).resolve().parent.parent.parent / "collective_autogen"
script = sdk_scripts / "eval_via_collective.py"
if script.exists():
    coll = script.resolve().parent.parent.parent / "collective_autogen"
    check("eval_via_collective.py: COLL resolves", coll.is_dir(),
          f"computed {coll}")
    gold = coll / "eval" / "ground_truth"
    check("eval_via_collective.py: GOLD_DIR resolves", gold.is_dir(),
          f"computed {gold}")
else:
    check("eval_via_collective.py present", False)

# failure_mode_analysis.py:
#   COLLECTIVE = Path(__file__).resolve().parent.parent        # → collective_autogen/
#   APPROACHES_ROOT = COLLECTIVE.parent                        # → approaches/
#   SDK_AGENT = APPROACHES_ROOT / "single_agent_sdk"
fma = ROOT / "approaches" / "collective_autogen" / "eval" / "failure_mode_analysis.py"
if fma.exists():
    collective = fma.resolve().parent.parent
    approaches_root = collective.parent
    sdk_agent = approaches_root / "single_agent_sdk"
    check("failure_mode_analysis.py: COLLECTIVE resolves", collective.is_dir(),
          f"computed {collective}")
    check("failure_mode_analysis.py: SDK_AGENT resolves", sdk_agent.is_dir(),
          f"computed {sdk_agent}")
else:
    check("failure_mode_analysis.py present", False)


# --------------------------------------------------------------------------
# 2. Compile check on every .py under approaches/
# --------------------------------------------------------------------------
section("Every .py under approaches/ parses")

py_files = list((ROOT / "approaches").rglob("*.py"))
# Exclude venv-like or build artifacts that may have slipped in
py_files = [p for p in py_files if ".venv" not in p.parts and "__pycache__" not in p.parts]

syntax_errors: list[tuple[Path, str]] = []
for p in py_files:
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
        compile(src, str(p), "exec")
    except SyntaxError as e:
        syntax_errors.append((p, f"line {e.lineno}: {e.msg}"))
    except Exception as e:  # noqa: BLE001
        syntax_errors.append((p, f"{type(e).__name__}: {e}"))

if not syntax_errors:
    check(f"{len(py_files)} files compile cleanly", True)
else:
    check(f"{len(py_files)} files compile cleanly", False,
          f"{len(syntax_errors)} files failed to parse")
    for p, err in syntax_errors[:10]:
        print(f"      {p.relative_to(ROOT)}: {err}")
    if len(syntax_errors) > 10:
        print(f"      ... and {len(syntax_errors) - 10} more")


# --------------------------------------------------------------------------
# 3. The modules the rewritten scripts import exist on disk
# --------------------------------------------------------------------------
section("Imported modules exist where the new paths point")

# From eval_via_collective.py and append_to_eval_xlsx.py:
#   from eval.metrics import evaluate
# Resolves under COLL = approaches/collective_autogen/
coll = ROOT / "approaches" / "collective_autogen"
check("collective_autogen/eval/__init__.py present",
      (coll / "eval" / "__init__.py").is_file())
check("collective_autogen/eval/metrics.py present",
      (coll / "eval" / "metrics.py").is_file())

# From append_to_eval_xlsx.py: `from eval_via_collective import ...`
# sys.path.insert(0, str(ROOT / "scripts")) — so the importable module is
# approaches/single_agent_sdk/scripts/eval_via_collective.py
check("single_agent_sdk/scripts/eval_via_collective.py present",
      (ROOT / "approaches" / "single_agent_sdk" / "scripts"
       / "eval_via_collective.py").is_file())

# From failure_mode_analysis.py:
#   from eval.metrics import (evaluate, _canonical, _canonical_no_stereo)
#   from scripts.eval_via_collective import figure_extraction_to_record
# scripts/ here is under SDK_AGENT, i.e. single_agent_sdk/scripts/
check("single_agent_sdk/scripts/ on path for failure_mode_analysis",
      (ROOT / "approaches" / "single_agent_sdk" / "scripts"
       / "eval_via_collective.py").is_file())

# AST inspection: confirm the imports we expect actually appear in the rewritten files
expected_imports = {
    "approaches/single_agent_sdk/scripts/eval_via_collective.py": "eval.metrics",
    "approaches/single_agent_sdk/scripts/append_to_eval_xlsx.py": "eval.metrics",
    "approaches/collective_autogen/eval/failure_mode_analysis.py": "eval.metrics",
}
for rel, mod in expected_imports.items():
    p = ROOT / rel
    if not p.exists():
        check(f"{rel} still imports {mod}", False, "file missing")
        continue
    tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    check(f"{rel} still imports {mod}", mod in imports,
          f"imports={sorted(imports)[:5]}…")


# --------------------------------------------------------------------------
# 4. Smoke-check: the path constants resolve to dirs that contain expected files
# --------------------------------------------------------------------------
section("Resolved COLL/SDK_AGENT dirs contain the files those scripts read")

# plot_cost_quality.py reads COLL / "eval" / "results" / "eval_summary.xlsx"
xlsx = coll / "eval" / "results" / "eval_summary.xlsx"
check("collective_autogen/eval/results/eval_summary.xlsx exists",
      xlsx.is_file(),
      "needed by plot_cost_quality.py / plot_aggregate_heatmap.py")

# failure_mode_analysis.py default arg: --bench-root ../single_agent_sdk/benchmark_runs
bench_root = (ROOT / "approaches" / "collective_autogen" / "eval"
              / "../../single_agent_sdk/benchmark_runs").resolve()
check("../../single_agent_sdk/benchmark_runs reachable from eval/",
      bench_root.is_dir(),
      f"resolved {bench_root}")

# Symlink chain: chemeagle_gemini/molnextr.pth → baselines/chemeagle/molnextr.pth → models/molnextr.pth
mp = ROOT / "approaches" / "chemeagle_gemini" / "molnextr.pth"
check("chemeagle_gemini/molnextr.pth resolves through symlink chain",
      mp.exists() and mp.resolve() == (ROOT / "models" / "molnextr.pth").resolve(),
      f"resolved to {mp.resolve()}")


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print(f"\n{BOLD}Result:{RESET} {passed} passed, {len(failed)} failed")
if failed:
    print("\nFailures:")
    for f in failed:
        print(f"  - {f}")
    sys.exit(1)
sys.exit(0)
