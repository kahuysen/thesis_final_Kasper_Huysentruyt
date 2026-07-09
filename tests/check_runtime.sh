#!/usr/bin/env bash
# Runtime smoke test: each venv's Python actually runs, can byte-compile the
# project's main entrypoint, and can import a few key project dependencies.
#
# This is the "truly test it" check — it goes beyond structure + AST and
# actually exercises the venv ↔ source ↔ models wiring on this machine.
#
# Run: bash tests/check_runtime.sh
#
# Exits non-zero if any check fails. Requires the .venv* symlinks set up
# (see top-level README). On a fresh machine without those venvs, expect
# every check in this file to fail — recreate them from each project's
# requirements before retrying.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pass=0
fail=0
fails=()

section() { printf "\n\033[1m%s\033[0m\n" "$1"; }
ok()   { pass=$((pass + 1)); printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad()  { fail=$((fail + 1)); fails+=("$1"); printf "  \033[31m✗\033[0m %s\n" "$1"; }

# ---- helpers -------------------------------------------------------------

# venv_python <label> <venv-path>
# checks bin/python exists & prints a version; sets VENV_PY if usable.
venv_python() {
  local label="$1" venv="$2"
  VENV_PY=""
  if [ ! -e "$venv/bin/python" ]; then
    bad "$label: $venv/bin/python missing"
    return 1
  fi
  local ver
  if ! ver=$("$venv/bin/python" --version 2>&1); then
    bad "$label: python not runnable ($ver)"
    return 1
  fi
  ok "$label: $ver"
  VENV_PY="$venv/bin/python"
}

# compile_with <venv-python> <source-file>
compile_with() {
  local py="$1" src="$2"
  if "$py" -m py_compile "$src" 2>/tmp/pycompile.err; then
    ok "py_compile $src"
  else
    bad "py_compile $src failed: $(tr -d '\n' < /tmp/pycompile.err | head -c 200)"
  fi
}

# import_check <venv-python> <module> [description]
import_check() {
  local py="$1"
  local mod="$2"
  local desc="${3-}"
  [ -z "$desc" ] && desc="import $mod"
  if "$py" -c "import $mod" >/dev/null 2>/tmp/import.err; then
    ok "$desc"
  else
    bad "$desc — $(tr -d '\n' < /tmp/import.err | head -c 200)"
  fi
}

# ---- single_agent_sdk ----------------------------------------------------

section "single_agent_sdk (.venv)"
if venv_python ".venv" "approaches/single_agent_sdk/.venv"; then
  PY="$VENV_PY"
  compile_with "$PY" approaches/single_agent_sdk/cli.py
  compile_with "$PY" approaches/single_agent_sdk/smoketest.py
  import_check "$PY" anthropic
  import_check "$PY" rdkit "import rdkit (chem toolkit)"
fi

section "single_agent_sdk (.venv-rxn-insight, subprocess driver)"
if venv_python ".venv-rxn-insight" "approaches/single_agent_sdk/.venv-rxn-insight"; then
  import_check "$VENV_PY" numpy
fi

# ---- collective_autogen --------------------------------------------------

section "collective_autogen (.venv)"
if venv_python ".venv" "approaches/collective_autogen/.venv"; then
  PY="$VENV_PY"
  compile_with "$PY" approaches/collective_autogen/main.py
  compile_with "$PY" approaches/collective_autogen/eval/metrics.py
  compile_with "$PY" approaches/collective_autogen/eval/failure_mode_analysis.py
  import_check "$PY" autogen_agentchat "import autogen_agentchat"
  import_check "$PY" openai
fi

section "collective_autogen secondary venvs"
if venv_python ".venv-molnextr" "approaches/collective_autogen/.venv-molnextr"; then
  import_check "$VENV_PY" torch
fi
if venv_python ".venv-openchemie" "approaches/collective_autogen/.venv-openchemie"; then
  import_check "$VENV_PY" torch
fi
if venv_python ".venv-rxninsight" "approaches/collective_autogen/.venv-rxninsight"; then
  import_check "$VENV_PY" numpy
fi

# ---- chemeagle (baseline) + chemeagle_gemini (approach) ------------------

section "chemeagle (.venv-chemeagle) via baselines/"
if venv_python ".venv-chemeagle" "baselines/chemeagle/.venv-chemeagle"; then
  PY="$VENV_PY"
  compile_with "$PY" baselines/chemeagle/main.py
  import_check "$PY" torch
fi

section "chemeagle_gemini reuses baselines/chemeagle/.venv-chemeagle"
if venv_python ".venv-chemeagle (via approach symlink)" "approaches/chemeagle_gemini/.venv-chemeagle"; then
  PY="$VENV_PY"
  compile_with "$PY" approaches/chemeagle_gemini/main.py
  compile_with "$PY" approaches/chemeagle_gemini/_gemini.py
fi

# ---- cross-cutting: the symlink chain resolves to a real model ----------

section "Model files reach through the symlink chain"
if [ -f "models/molnextr.pth" ] && [ -f "approaches/chemeagle_gemini/molnextr.pth" ]; then
  s1=$(stat -f %z models/molnextr.pth 2>/dev/null || stat -c %s models/molnextr.pth)
  s2=$(stat -f %z -L approaches/chemeagle_gemini/molnextr.pth 2>/dev/null || \
       stat -c %s approaches/chemeagle_gemini/molnextr.pth)
  if [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ]; then
    ok "chemeagle_gemini/molnextr.pth → models/molnextr.pth (size $s1)"
  else
    bad "molnextr.pth chain sizes differ: $s1 vs $s2"
  fi
else
  bad "molnextr.pth not found at expected path"
fi

# ---- cleanup: py_compile leaves __pycache__ next to each source --------

section "Cleanup (removing __pycache__ created by py_compile)"
removed=$(find . -type d -name __pycache__ \
            -not -path '*/.venv*' -not -path '*/.git/*' 2>/dev/null | wc -l | tr -d ' ')
find . -type d -name __pycache__ \
  -not -path '*/.venv*' -not -path '*/.git/*' \
  -exec rm -rf {} + 2>/dev/null
ok "removed $removed __pycache__ dirs"

# ---- summary -------------------------------------------------------------

printf "\n\033[1mResult:\033[0m %d passed, %d failed\n" "$pass" "$fail"
if [ "$fail" -gt 0 ]; then
  printf "\nFailures:\n"
  for f in "${fails[@]}"; do printf "  - %s\n" "$f"; done
  exit 1
fi
