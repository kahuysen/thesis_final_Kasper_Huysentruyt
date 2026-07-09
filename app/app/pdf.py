"""PDF processing orchestrator (review-before-extract flow).

Two-phase pipeline:

  1. **Detect** — `subprocess_drivers/visualheist_runner.py` runs in
     `.venv-chemeagle` (Python 3.10 + torch + transformers + a cached
     VisualHeist Florence-2 fine-tune). It rasterizes the PDF with
     pypdfium2 and detects figure/table bboxes per page, saving each
     crop to `pdf_jobs/<job_id>/figures/page_NNN_fig_NN.png`.

  2. **Submit** — the UI shows the detected crops; the user picks which
     ones to extract; `submit_figures()` creates one child run per
     selected crop and starts the regular Claude vision agent on it.

Persistent layout:

    pdf_jobs/<job_id>/
        source.pdf
        figures/page_*.png     raw crops emitted by VisualHeist
        job.json               {status, total_pages, figures, child_runs, errors}

    runs/<child_id>/           created at submit time
        input.png              copy of the chosen crop
        pdf_source.json        {pdf_job_id, page, bbox, figure_id}
        ...                    standard run artifacts after extraction
"""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from . import folders as folder_store
from .runs import RUNS_DIR

ROOT = Path(__file__).resolve().parent.parent
PDF_JOBS_DIR = ROOT / "pdf_jobs"
VENV_PYTHON = ROOT / ".venv-chemeagle" / "bin" / "python3"
RUNNER_SCRIPT = ROOT / "subprocess_drivers" / "visualheist_runner.py"

# Cap concurrent agent extractions across all PDF jobs to avoid rate limits.
EXTRACTION_SEMAPHORE = threading.Semaphore(2)


def available() -> bool:
    return VENV_PYTHON.exists() and RUNNER_SCRIPT.exists()


def new_job() -> tuple[str, Path]:
    job_id = uuid.uuid4().hex[:12]
    d = PDF_JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(exist_ok=True)
    return job_id, d


def job_path(job_id: str) -> Optional[Path]:
    if not job_id or "/" in job_id or ".." in job_id:
        return None
    p = PDF_JOBS_DIR / job_id
    return p if p.is_dir() else None


def _write_job_state(job_dir: Path, state: dict) -> None:
    (job_dir / "job.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def read_job_state(job_dir: Path) -> dict:
    p = job_dir / "job.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _figure_id(page: int, index: int) -> str:
    return f"p{page:03d}_f{index:02d}"


def _stream_runner(pdf_path: Path, out_dir: Path, model_size: str):
    """Generator yielding parsed JSON event dicts from the VisualHeist subprocess."""
    payload = json.dumps({
        "pdf_path":   str(pdf_path),
        "out_dir":    str(out_dir),
        "model_size": model_size,
    })
    proc = subprocess.Popen(
        [str(VENV_PYTHON), str(RUNNER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin and proc.stdout and proc.stderr
    proc.stdin.write(payload)
    proc.stdin.close()

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"event": "log", "line": line}
    finally:
        proc.wait(timeout=10)
        if proc.returncode not in (0, None):
            try:
                err = proc.stderr.read()
            except Exception:
                err = ""
            if err:
                yield {"event": "stderr", "text": err[-4000:]}


# ---------- phase 1: detection ----------

def detect_figures(
    job_id: str,
    job_dir: Path,
    pdf_path: Path,
    model_size: str,
    on_event,
) -> None:
    """Run VisualHeist on the PDF. No child runs are created.

    Emits via `on_event`:
        start              {total_pages}
        loading_model      {model_size}
        model_ready
        page_done          {page, figure_count}
        figure_detected    {figure_id, page, figure_index, bbox}
        detection_complete {figures: [...]}
        error              {message}
    """
    state: dict = {
        "job_id":   job_id,
        "created":  time.time(),
        "status":   "detecting",
        "pdf_name": pdf_path.name,
        "total_pages": None,
        "figures":  [],        # [{figure_id, page, index, bbox, path}]
        "child_runs": [],
        "errors":   [],
    }
    _write_job_state(job_dir, state)

    try:
        for ev in _stream_runner(pdf_path, job_dir / "figures", model_size):
            kind = ev.get("event")
            if kind == "start":
                state["total_pages"] = ev.get("total_pages")
                _write_job_state(job_dir, state)
                on_event(ev)
            elif kind == "loading_model":
                on_event(ev)
            elif kind == "model_ready":
                on_event(ev)
            elif kind == "page_done":
                page = ev.get("page")
                figs_in = ev.get("figures") or []
                for fig in figs_in:
                    fid = _figure_id(fig["page"], fig["index"])
                    rel_path = f"figures/{Path(fig['path']).name}"
                    record = {
                        "figure_id":    fid,
                        "page":         fig["page"],
                        "figure_index": fig["index"],
                        "bbox":         fig["bbox"],
                        "path":         rel_path,
                    }
                    state["figures"].append(record)
                    on_event({"event": "figure_detected", **record})
                _write_job_state(job_dir, state)
                on_event({
                    "event":        "page_done",
                    "page":         page,
                    "figure_count": len(figs_in),
                })
            elif kind == "error":
                state["status"] = "error"
                state["errors"].append(ev.get("message", "unknown error"))
                _write_job_state(job_dir, state)
                on_event(ev)
                on_event({"event": "_eof"})
                return
            elif kind == "stderr":
                state["errors"].append(ev.get("text", ""))
                _write_job_state(job_dir, state)
            elif kind == "done":
                # Subprocess finished — wait for the loop to exit naturally.
                pass
    except FileNotFoundError as exc:
        on_event({
            "event":   "error",
            "message": (f"VisualHeist runtime not configured: {exc}. "
                        "Symlink .venv-chemeagle and ensure the runner script is present."),
        })
        state["status"] = "error"
        state["errors"].append(str(exc))
        _write_job_state(job_dir, state)
        on_event({"event": "_eof"})
        return

    state["status"] = "awaiting_selection"
    _write_job_state(job_dir, state)
    on_event({"event": "detection_complete", "figures": state["figures"]})


# ---------- phase 2: submit selected figures for extraction ----------

class SubmitError(Exception):
    pass


def submit_figures(
    job_id: str,
    job_dir: Path,
    figure_ids: list[str],
    extract_model: Optional[str],
    extract_provider: Optional[str],
    extract_fn,                  # callable(child_run_dir, model, provider, queue.Queue)
    on_event,
    folder_name: Optional[str] = None,
) -> list[dict]:
    """Spin up one child run per selected figure and start extraction.

    Returns: [{figure_id, child_run_id}, ...] for the submitted figures.
    Raises SubmitError if already submitted or no valid figure_ids.

    Emits via `on_event`:
        submit_started       {child_count}
        figure_selected      {figure_id, child_run_id}
        child_started        {child_run_id}
        child_progress       {child_run_id, step, tool, ok}
        child_complete       {child_run_id, n_reactions}
        child_error          {child_run_id, message}
        complete             {n_figures, n_done, n_failed}
    """
    state = read_job_state(job_dir)
    if state.get("status") not in ("awaiting_selection", "extracting", "complete_with_errors"):
        raise SubmitError(f"can't submit when status is {state.get('status')!r}")
    if state.get("child_runs"):
        raise SubmitError("figures already submitted for this PDF job")

    by_fid = {f["figure_id"]: f for f in state.get("figures", [])}
    selected = [by_fid[fid] for fid in figure_ids if fid in by_fid]
    if not selected:
        raise SubmitError("no valid figure_ids selected")

    # Create a folder (if requested) and remember its id so every child
    # run materialized below gets filed into it automatically.
    folder_id: Optional[str] = None
    if folder_name and folder_name.strip():
        try:
            folder = folder_store.create_folder(folder_name.strip())
            folder_id = folder["id"]
            state["folder_id"]   = folder_id
            state["folder_name"] = folder["name"]
            on_event({
                "event":       "folder_created",
                "folder_id":   folder_id,
                "folder_name": folder["name"],
            })
        except ValueError as exc:
            # Bad folder name — surface a warning but keep going (runs land unfiled).
            on_event({"event": "folder_error", "message": str(exc)})

    state["status"] = "extracting"
    state["child_runs"] = []
    _write_job_state(job_dir, state)
    on_event({"event": "submit_started", "child_count": len(selected)})

    threads: list[threading.Thread] = []
    pairs: list[dict] = []

    def _mark_child(child_id: str, status: str, **extra) -> None:
        for cr in state["child_runs"]:
            if cr["child_run_id"] == child_id:
                cr["status"] = status
                cr.update(extra)
                break
        _write_job_state(job_dir, state)

    def _spawn(fig: dict) -> dict:
        child_id = uuid.uuid4().hex[:12]
        child_dir = RUNS_DIR / child_id
        child_dir.mkdir(parents=True, exist_ok=True)
        src = job_dir / fig["path"]
        shutil.copy(src, child_dir / "input.png")
        (child_dir / "pdf_source.json").write_text(json.dumps({
            "pdf_job_id":   job_id,
            "figure_id":    fig["figure_id"],
            "page":         fig["page"],
            "figure_index": fig["figure_index"],
            "bbox":         fig["bbox"],
        }, indent=2), encoding="utf-8")

        cr = {
            "child_run_id": child_id,
            "figure_id":    fig["figure_id"],
            "page":         fig["page"],
            "figure_index": fig["figure_index"],
            "bbox":         fig["bbox"],
            "status":       "queued",
        }
        state["child_runs"].append(cr)
        _write_job_state(job_dir, state)
        if folder_id is not None:
            try:
                folder_store.assign_run(child_id, folder_id)
            except Exception:
                # Folder went missing between create and assign — best effort,
                # the run is still usable, just unfiled.
                pass
        on_event({"event": "figure_selected",
                  "figure_id": fig["figure_id"],
                  "child_run_id": child_id})

        def _runner():
            with EXTRACTION_SEMAPHORE:
                _mark_child(child_id, "running")
                on_event({"event": "child_started", "child_run_id": child_id})

                step_count = 0
                child_q: "queue.Queue[dict]" = queue.Queue()
                ext_thread = threading.Thread(
                    target=extract_fn,
                    args=(child_dir, extract_model, extract_provider, child_q),
                    daemon=True,
                )
                ext_thread.start()

                n_reactions = None
                error_msg = None
                while True:
                    try:
                        ev = child_q.get(timeout=600)
                    except queue.Empty:
                        error_msg = "child extraction timed out"
                        break
                    if ev.get("event") == "_eof":
                        break
                    kind = ev.get("event")
                    if kind == "step_done":
                        step_count += 1
                        on_event({"event": "child_progress",
                                  "child_run_id": child_id,
                                  "step": step_count,
                                  "tool": ev.get("tool"),
                                  "ok":   ev.get("ok")})
                    elif kind == "done":
                        n_reactions = ev.get("n_reactions")
                    elif kind == "error":
                        error_msg = ev.get("message") or "extraction error"
                ext_thread.join(timeout=5)

                if error_msg:
                    _mark_child(child_id, "error", error=error_msg)
                    on_event({"event": "child_error", "child_run_id": child_id, "message": error_msg})
                else:
                    _mark_child(child_id, "done", n_reactions=n_reactions)
                    on_event({"event": "child_complete", "child_run_id": child_id, "n_reactions": n_reactions})

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        threads.append(t)
        return cr

    for fig in selected:
        cr = _spawn(fig)
        pairs.append({"figure_id": fig["figure_id"], "child_run_id": cr["child_run_id"]})

    def _wait_and_finalize():
        for t in threads:
            t.join()
        s = read_job_state(job_dir)
        n_total  = len(s.get("child_runs", []))
        n_done   = sum(1 for c in s.get("child_runs", []) if c.get("status") == "done")
        n_failed = sum(1 for c in s.get("child_runs", []) if c.get("status") == "error")
        s["status"] = "complete" if not s.get("errors") else "complete_with_errors"
        _write_job_state(job_dir, s)
        on_event({
            "event":     "complete",
            "n_figures": n_total,
            "n_done":    n_done,
            "n_failed":  n_failed,
        })
        on_event({"event": "_eof"})

    threading.Thread(target=_wait_and_finalize, daemon=True).start()
    return pairs
