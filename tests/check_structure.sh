#!/usr/bin/env bash
# Verify the reorganised tree still has the layout, files, and symlinks it should.
# Run from anywhere: `bash tests/check_structure.sh`
# Exits non-zero if any check fails. No external deps beyond find/readlink/test.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pass=0
fail=0
fails=()

check() {
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then
    pass=$((pass + 1))
    printf "  \033[32m✓\033[0m %s\n" "$desc"
  else
    fail=$((fail + 1))
    fails+=("$desc")
    printf "  \033[31m✗\033[0m %s\n" "$desc"
  fi
}

section() { printf "\n\033[1m%s\033[0m\n" "$1"; }

section "Top-level directories"
for d in approaches baselines models data outputs; do
  check "$d/ exists" test -d "$d"
done
check "top-level README.md exists" test -f README.md
check "side_projects/ absent (deleted by user)" test ! -e side_projects

section "Approaches"
for sub in single_agent_sdk collective_autogen chemeagle_gemini; do
  check "approaches/$sub/ exists" test -d "approaches/$sub"
done
check "single_agent_sdk has cli.py"      test -f approaches/single_agent_sdk/cli.py
check "single_agent_sdk has pipeline/"   test -d approaches/single_agent_sdk/pipeline
check "single_agent_sdk has README.md"   test -f approaches/single_agent_sdk/README.md
check "collective_autogen has main.py"   test -f approaches/collective_autogen/main.py
check "collective_autogen has CLAUDE.md" test -f approaches/collective_autogen/CLAUDE.md
check "collective_autogen has eval/"     test -d approaches/collective_autogen/eval
check "chemeagle_gemini has main.py"     test -f approaches/chemeagle_gemini/main.py
check "chemeagle_gemini has _gemini.py"  test -f approaches/chemeagle_gemini/_gemini.py

section "Baselines (verbatim)"
for sub in chemeagle rxn_insight maarten_chemeagle_gemini; do
  check "baselines/$sub/ exists" test -d "baselines/$sub"
done
check "chemeagle has main.py"            test -f baselines/chemeagle/main.py
check "rxn_insight has pyproject.toml"   test -f baselines/rxn_insight/pyproject.toml
check "rxn_insight has src/"             test -d baselines/rxn_insight/src
check "maarten has ChemEagle_Hybrid/"    test -d baselines/maarten_chemeagle_gemini/ChemEagle_Hybrid
check "maarten has Gen-Rxn-INSIGHT/"     test -d baselines/maarten_chemeagle_gemini/Gen-Rxn-INSIGHT

section "Shared models/"
for w in ner.ckpt rxn.ckpt moldet.ckpt corefdet.ckpt molnextr.pth \
         biobert-large-cased cre_models_v0.1 Tesseract-OCR; do
  check "models/$w exists" test -e "models/$w"
done
check "models/README.md exists" test -f models/README.md

section "Data"
check "data/benchmark/ exists"        test -d data/benchmark
check "data/test_papers/ exists"      test -d data/test_papers
check "data/smirks_db.json exists"    test -f data/smirks_db.json

section "Outputs"
check "outputs/README.md exists"                  test -f outputs/README.md
check "outputs/single_agent_sdk/benchmark_runs/"  test -d outputs/single_agent_sdk/benchmark_runs
check "outputs/single_agent_sdk/results/"         test -d outputs/single_agent_sdk/results
check "web_results/ absent (deleted by user)"     test ! -e outputs/single_agent_sdk/web_results
check "web_uploads/ absent (deleted by user)"     test ! -e outputs/single_agent_sdk/web_uploads
check "outputs/collective_autogen/runs/"          test -d outputs/collective_autogen/runs
check "outputs/collective_autogen/cache/"         test -d outputs/collective_autogen/cache
check "outputs/collective_autogen/figures/"       test -d outputs/collective_autogen/figures
check "outputs/collective_autogen/databank/"      test -d outputs/collective_autogen/databank


section "No broken symlinks anywhere (excluding .claude/ + .git/)"
broken=$(find . -xtype l -not -path '*/.git/*' -not -path '*/.claude/*' 2>/dev/null)
if [ -z "$broken" ]; then
  pass=$((pass + 1))
  printf "  \033[32m✓\033[0m all symlinks resolve\n"
else
  fail=$((fail + 1))
  fails+=("broken symlinks present")
  printf "  \033[31m✗\033[0m broken symlinks:\n%s\n" "$broken"
fi

section "Symlink targets are relative (portable; excluding .venv/.claude/.git)"
# Absolute symlinks are allowed for .venv* (they intentionally point at originals).
unexpected=""
while IFS= read -r link; do
  tgt=$(readlink "$link")
  case "$tgt" in
    /*) unexpected="$unexpected"$'\n'"$link -> $tgt" ;;
  esac
done < <(find . -type l \
            -not -path '*/.git/*' \
            -not -path '*/.claude/*' \
            -not -name '.venv' -not -name '.venv-*' \
            2>/dev/null)

if [ -z "$unexpected" ]; then
  pass=$((pass + 1))
  printf "  \033[32m✓\033[0m no unexpected absolute-path symlinks\n"
else
  fail=$((fail + 1))
  fails+=("absolute-path symlinks present")
  printf "  \033[31m✗\033[0m these symlinks still use absolute paths:%s\n" "$unexpected"
fi

section "chemeagle_gemini symlinks point to baselines/chemeagle/"
mis=0
for l in approaches/chemeagle_gemini/chemiener \
         approaches/chemeagle_gemini/chemietoolkit \
         approaches/chemeagle_gemini/molnextr.pth \
         approaches/chemeagle_gemini/biobert-large-cased \
         approaches/chemeagle_gemini/ner.ckpt; do
  if [ -L "$l" ]; then
    tgt=$(readlink "$l")
    case "$tgt" in
      ../../baselines/chemeagle/*) ;;
      *) mis=1; printf "  \033[31m✗\033[0m %s -> %s (expected ../../baselines/chemeagle/...)\n" "$l" "$tgt" ;;
    esac
  fi
done
if [ $mis -eq 0 ]; then
  pass=$((pass + 1))
  printf "  \033[32m✓\033[0m sampled chemeagle_gemini links point at baselines/chemeagle/\n"
else
  fail=$((fail + 1))
  fails+=("chemeagle_gemini link targets wrong")
fi

section "baselines/chemeagle/ weights are symlinks into models/"
for w in ner.ckpt rxn.ckpt moldet.ckpt corefdet.ckpt molnextr.pth \
         biobert-large-cased cre_models_v0.1 Tesseract-OCR; do
  p="baselines/chemeagle/$w"
  if [ -L "$p" ]; then
    tgt=$(readlink "$p")
    case "$tgt" in
      ../../models/*) pass=$((pass + 1)); printf "  \033[32m✓\033[0m %s -> %s\n" "$w" "$tgt" ;;
      *) fail=$((fail + 1)); fails+=("$w bad target"); printf "  \033[31m✗\033[0m %s -> %s\n" "$w" "$tgt" ;;
    esac
  else
    fail=$((fail + 1)); fails+=("$w not a symlink")
    printf "  \033[31m✗\033[0m baselines/chemeagle/%s is not a symlink\n" "$w"
  fi
done

section "Output compat symlinks (project → outputs/)"
for pair in \
  "approaches/single_agent_sdk/benchmark_runs|../../outputs/single_agent_sdk/benchmark_runs" \
  "approaches/single_agent_sdk/results|../../outputs/single_agent_sdk/results" \
  "approaches/collective_autogen/runs|../../outputs/collective_autogen/runs" \
  "approaches/collective_autogen/cache|../../outputs/collective_autogen/cache" \
  "approaches/collective_autogen/figures|../../outputs/collective_autogen/figures" \
  "approaches/collective_autogen/databank|../../outputs/collective_autogen/databank" \
  "approaches/collective_autogen/eval/Benchmark_kasper_GT3_Maarten|../../../data/benchmark"
do
  src=${pair%|*}; want=${pair#*|}
  if [ -L "$src" ]; then
    got=$(readlink "$src")
    if [ "$got" = "$want" ]; then
      pass=$((pass + 1)); printf "  \033[32m✓\033[0m %s\n" "$src"
    else
      fail=$((fail + 1)); fails+=("$src wrong target")
      printf "  \033[31m✗\033[0m %s -> %s (expected %s)\n" "$src" "$got" "$want"
    fi
  else
    fail=$((fail + 1)); fails+=("$src not a symlink")
    printf "  \033[31m✗\033[0m %s missing or not a symlink\n" "$src"
  fi
done

section "Excluded items are absent"
for n in thesis_report_sdk_agent IDEAS.md rxn_insight_documentation \
         failure_modes_raw failure_modes_sdk_variants \
         UI_benchmark ReactionSeek "deepseek results" \
         Tesseract-OCR.zip biobert-large-cased.zip cre_models_v0.1.zip; do
  hits=$(find . -name "*${n}*" -not -path '*/.git/*' 2>/dev/null | head -1)
  if [ -z "$hits" ]; then
    pass=$((pass + 1)); printf "  \033[32m✓\033[0m absent: %s\n" "$n"
  else
    fail=$((fail + 1)); fails+=("$n still present")
    printf "  \033[31m✗\033[0m found: %s (at %s)\n" "$n" "$hits"
  fi
done

section "Virtualenvs reachable (symlinked to originals on this machine)"
for pair in \
  "approaches/single_agent_sdk/.venv" \
  "approaches/single_agent_sdk/.venv-rxn-insight" \
  "approaches/collective_autogen/.venv" \
  "approaches/collective_autogen/.venv-molnextr" \
  "approaches/collective_autogen/.venv-openchemie" \
  "approaches/collective_autogen/.venv-rxninsight" \
  "baselines/chemeagle/.venv-chemeagle" \
  "approaches/chemeagle_gemini/.venv-chemeagle"
do
  if [ -x "$pair/bin/python" ]; then
    pass=$((pass + 1)); printf "  \033[32m✓\033[0m %s/bin/python exists\n" "$pair"
  else
    fail=$((fail + 1)); fails+=("$pair missing python")
    printf "  \033[31m✗\033[0m %s/bin/python missing\n" "$pair"
  fi
done

pyc=$(find . -name __pycache__ -not -path '*/.git/*' -not -path '*/.venv*' 2>/dev/null | head -1)
if [ -z "$pyc" ]; then
  pass=$((pass + 1)); printf "  \033[32m✓\033[0m no __pycache__ in source tree\n"
else
  fail=$((fail + 1)); fails+=("__pycache__ present")
  printf "  \033[31m✗\033[0m found __pycache__: %s\n" "$pyc"
fi

printf "\n\033[1mResult:\033[0m %d passed, %d failed\n" "$pass" "$fail"
if [ "$fail" -gt 0 ]; then
  printf "\nFailures:\n"
  for f in "${fails[@]}"; do printf "  - %s\n" "$f"; done
  exit 1
fi
