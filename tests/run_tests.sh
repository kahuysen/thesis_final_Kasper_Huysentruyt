#!/usr/bin/env bash
# Run all reorg sanity tests. From anywhere:
#   bash tests/run_tests.sh
# Returns 0 iff every check passed.

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

fail=0

printf "\033[1m=== Structure check ===\033[0m\n"
bash "$HERE/check_structure.sh" || fail=1

printf "\n\033[1m=== Path-resolution & compile check ===\033[0m\n"
python3 "$HERE/check_paths.py" || fail=1

printf "\n\033[1m=== Runtime smoke check (uses each venv) ===\033[0m\n"
bash "$HERE/check_runtime.sh" || fail=1

printf "\n"
if [ "$fail" -eq 0 ]; then
  printf "\033[32mAll tests passed.\033[0m\n"
else
  printf "\033[31mOne or more test files reported failures.\033[0m\n"
  exit 1
fi
