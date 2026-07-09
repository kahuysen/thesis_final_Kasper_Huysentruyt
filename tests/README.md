# Reorg sanity tests

These tests verify the **reorganised layout** itself — directory structure, symlink integrity, and that the path expressions I edited in 5 source files still resolve. They do **not** run any of the actual pipelines (no LLM calls, no virtualenvs needed).

## Run

```bash
bash tests/run_tests.sh
```

Or each part individually:

```bash
bash tests/check_structure.sh   # tree, files, symlinks, exclusions
python3 tests/check_paths.py    # path constants + .py compile + import targets
```

System Python 3.8+ is enough; no `pip install` required.

## What each file checks

**`check_structure.sh`** — directory and symlink integrity:
- All top-level dirs (`approaches/`, `baselines/`, `models/`, …) exist.
- Each project has its main entrypoints (`main.py`, `app.py`, `cli.py`, …).
- All eight shared model weights present under `models/`.
- `find -xtype l` reports no broken symlinks.
- Every symlink under the tree uses a **relative** target (portable).
- `chemeagle_gemini/` links point at `../../baselines/chemeagle/...`.
- `baselines/chemeagle/{ner.ckpt, …}` are symlinks to `../../models/...`.
- The 7 compat symlinks (project folder → `outputs/...` and `data/benchmark`) all resolve to the right place.
- 11 excluded items (`UI_benchmark`, `ReactionSeek`, `deepseek results`, the 3 zips, 3 write-ups, 2 failure-mode reports) are absent.
- No `.venv*` or `__pycache__` leaked into the shared copy.

**`check_paths.py`** — code-side resolution:
- The path expressions I rewrote (`COLL = ROOT.parent / "collective_autogen"` etc.) resolve to real directories in all 5 edited files.
- Every `.py` under `approaches/` parses with `compile()` (catches accidental syntax breakage from the edits).
- The modules those scripts import (`eval.metrics`, `scripts.eval_via_collective`) actually exist at the paths the rewritten constants point to.
- AST scan confirms the expected `from eval.metrics import …` lines still appear in the rewritten files.
- Resolved `COLL` dir contains the data file `plot_cost_quality.py` reads (`eval/results/eval_summary.xlsx`).
- The two-hop symlink chain `chemeagle_gemini/molnextr.pth → baselines/chemeagle/molnextr.pth → models/molnextr.pth` resolves to the file in `models/`.

## What these tests do NOT cover

- Actually running any pipeline (would need recreating the venvs from `requirements.txt`).
- Whether the supervised models still load correctly (would need PyTorch and ~12 GB RAM).
- API connectivity (Anthropic / OpenAI / Gemini keys, RDKit installs, etc.).

If you want end-to-end verification, recreate a venv in one project and run its smallest entrypoint, e.g.:

```bash
cd approaches/single_agent_sdk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python smoketest.py
```
