"""FastAPI app wrapping the single-SDK chemistry vision agent pipeline.

Endpoints:
    GET  /                            -> static/index.html
    GET  /api/health                  -> backend + capability info
    POST /api/runs                    -> upload one image, returns {run_id}
    GET  /api/runs/{id}/events        -> Server-Sent Events stream of agent steps
    GET  /api/runs/{id}/file/{name}   -> serve any artifact under runs/<id>/
    POST /api/runs/{id}/insight       -> optional Rxn-INSIGHT enrichment

Run locally:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import mimetypes
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from pipeline import config as pipeline_config
from pipeline.extractor import extract_figure_stream, DEFAULT_MODEL
from pipeline.flatten import figure_to_rows, write_csv
from pipeline.rxn_insight import (
    VENV_PYTHON as RXN_INSIGHT_VENV,
    analyze_extraction,
    extraction_to_insight_rows,
    write_insight_csv,
)
from pipeline.schema import FigureExtraction

from . import folders as folder_store
from . import pdf as pdf_jobs
from . import settings as settings_store
from .runs import RUNS_DIR, find_input_image, new_run, run_path, safe_child

ALLOWED_MIME_PREFIXES = ("image/",)
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="ReactionMiner", description="Chemistry figure → structured reactions.")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    try:
        provider = pipeline_config.current_provider()
        model = pipeline_config.default_model()
        backend = pipeline_config.describe()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        models = pipeline_config.available_models()
    except Exception:
        models = []

    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "backend": backend,
        "models": models,
        "rxn_insight_available": RXN_INSIGHT_VENV.exists(),
        "pdf_available": pdf_jobs.available(),
    }


# Dark-theme atom palette: light carbons/bonds, brightened heteroatoms so they
# stay recognizable on a dark panel. RGB tuples are 0..1.
_DARK_PALETTE = {
    -1: (0.88, 0.88, 0.92),   # default (incl. bonds between unlabeled atoms)
    0:  (0.70, 0.70, 0.75),   # dummy / wildcards
    1:  (0.88, 0.88, 0.92),   # H
    6:  (0.88, 0.88, 0.92),   # C
    7:  (0.55, 0.75, 1.00),   # N — lighter blue
    8:  (1.00, 0.45, 0.45),   # O — softer red
    9:  (0.55, 0.95, 0.55),   # F
    15: (1.00, 0.65, 0.40),   # P — orange
    16: (1.00, 0.85, 0.40),   # S — yellow
    17: (0.55, 0.95, 0.55),   # Cl
    35: (0.95, 0.65, 0.45),   # Br
    53: (0.85, 0.55, 1.00),   # I — purple
}


@app.get("/api/mol")
def mol_image(smiles: str, w: int = 220, h: int = 160) -> Response:
    """Render a single SMILES to a transparent-background PNG via RDKit."""
    if not smiles or len(smiles) > 1000:
        raise HTTPException(status_code=400, detail="invalid smiles")
    if not (32 <= w <= 800) or not (32 <= h <= 800):
        raise HTTPException(status_code=400, detail="size out of range")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail="unparseable smiles")

    AllChem.Compute2DCoords(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.padding = 0.08
    opts.clearBackground = False  # leave alpha transparent
    opts.updateAtomPalette(_DARK_PALETTE)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    return Response(
        content=drawer.GetDrawingText(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/runs")
def list_runs(limit: int = 100) -> list[dict]:
    """List previously executed runs (most-recent first).

    Each entry carries enough summary for the "My runs" picker to render a
    thumbnail + counts without re-parsing every artifact. `folder_id` is
    `None` for runs that haven't been filed.
    """
    if not RUNS_DIR.exists():
        return []

    assignments = folder_store.list_assignments()

    entries = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        try:
            mtime = d.stat().st_mtime
        except OSError:
            continue

        input_name = None
        for p in d.iterdir():
            if p.is_file() and p.name.startswith("input."):
                input_name = p.name
                break

        info: dict = {
            "run_id": d.name,
            "created": mtime,
            "input_name": input_name,
            "has_extraction": False,
            "has_insight": (d / "insight.json").exists(),
            "n_reactions": None,
            "figure_caption": None,
            "folder_id": assignments.get(d.name),
        }

        ext_file = d / "extraction.json"
        if ext_file.exists():
            info["has_extraction"] = True
            try:
                ex = json.loads(ext_file.read_text(encoding="utf-8"))
                info["n_reactions"] = len(ex.get("reactions") or [])
                cap = ex.get("figure_caption")
                if cap:
                    info["figure_caption"] = cap[:240]
            except Exception:
                pass

        entries.append(info)

    entries.sort(key=lambda x: x["created"], reverse=True)
    return entries[:max(1, min(int(limit), 500))]


# ---------- PDF jobs ----------

# Map of job_id -> queue.Queue[dict] used by the SSE endpoint to stream events
# emitted while a PDF job is being processed. We keep the queue around for a
# short while after completion so a late-joining SSE client can still receive
# the trailing events.
_PDF_QUEUES: dict[str, "queue.Queue[dict]"] = {}
_PDF_QUEUES_LOCK = threading.Lock()


def _get_pdf_queue(job_id: str) -> "queue.Queue[dict]":
    with _PDF_QUEUES_LOCK:
        q = _PDF_QUEUES.get(job_id)
        if q is None:
            q = queue.Queue()
            _PDF_QUEUES[job_id] = q
        return q


@app.post("/api/pdf_jobs")
async def create_pdf_job(
    image: UploadFile = File(...),
    model_size: str = "large",
) -> dict:
    """Upload a PDF and start VisualHeist figure detection.

    No extractions are kicked off yet — the caller reviews detected figures
    and then POSTs the selected `figure_id`s to `/api/pdf_jobs/{id}/submit`.
    """
    if not pdf_jobs.available():
        raise HTTPException(
            status_code=503,
            detail="PDF processing not configured: symlink .venv-chemeagle and ensure subprocess_drivers/visualheist_runner.py exists.",
        )

    ctype = (image.content_type or "").lower()
    name = (image.filename or "").lower()
    if "pdf" not in ctype and not name.endswith(".pdf"):
        raise HTTPException(status_code=415, detail=f"Expected a PDF (got {ctype!r}).")

    job_id, job_dir = pdf_jobs.new_job()
    pdf_path = job_dir / "source.pdf"
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    pdf_path.write_bytes(data)

    q = _get_pdf_queue(job_id)

    def on_event(ev: dict) -> None:
        q.put(ev)

    def _detect_thread():
        try:
            pdf_jobs.detect_figures(
                job_id=job_id,
                job_dir=job_dir,
                pdf_path=pdf_path,
                model_size=model_size,
                on_event=on_event,
            )
        except Exception as exc:
            on_event({"event": "error", "message": f"{type(exc).__name__}: {exc}"})

    threading.Thread(target=_detect_thread, daemon=True).start()
    return {"job_id": job_id, "kind": "pdf", "bytes": len(data), "pdf_name": image.filename}


@app.post("/api/pdf_jobs/{job_id}/submit")
async def pdf_job_submit(job_id: str, payload: dict) -> dict:
    """Submit a subset of detected figures for extraction.

    Body: { "figure_ids": ["p002_f01", ...], "model"?: str, "provider"?: str }
    """
    job_dir = pdf_jobs.job_path(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="pdf job not found")

    figure_ids = (payload or {}).get("figure_ids") or []
    if not isinstance(figure_ids, list) or not figure_ids:
        raise HTTPException(status_code=400, detail="figure_ids must be a non-empty list")

    model       = (payload or {}).get("model")
    provider    = (payload or {}).get("provider")
    folder_name = (payload or {}).get("folder_name")

    q = _get_pdf_queue(job_id)

    def on_event(ev: dict) -> None:
        q.put(ev)

    try:
        pairs = pdf_jobs.submit_figures(
            job_id=job_id,
            job_dir=job_dir,
            figure_ids=figure_ids,
            extract_model=model,
            extract_provider=provider,
            extract_fn=_run_pipeline,
            on_event=on_event,
            folder_name=folder_name,
        )
    except pdf_jobs.SubmitError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "pairs": pairs}


@app.get("/api/pdf_jobs/{job_id}")
def pdf_job_status(job_id: str) -> dict:
    job_dir = pdf_jobs.job_path(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="pdf job not found")
    return pdf_jobs.read_job_state(job_dir)


@app.get("/api/pdf_jobs/{job_id}/events")
def pdf_job_events(job_id: str):
    job_dir = pdf_jobs.job_path(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="pdf job not found")

    q = _get_pdf_queue(job_id)

    def gen():
        # If the job already finished, replay the saved state as a single
        # synthetic event so a late client doesn't see an empty stream.
        state = pdf_jobs.read_job_state(job_dir)
        if state.get("status") in ("complete", "complete_with_errors", "error"):
            yield f"data: {json.dumps({'event': 'state', 'state': state})}\n\n"
            return
        yield ": stream open\n\n"
        while True:
            ev = q.get()
            if ev.get("event") == "_eof":
                return
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/pdf_jobs/{job_id}/file/{name:path}")
def pdf_job_file(job_id: str, name: str):
    job_dir = pdf_jobs.job_path(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="pdf job not found")
    target = safe_child(job_dir, name)
    if target is None:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)


# ---------- settings ----------

@app.get("/api/settings")
def settings_get() -> dict:
    return settings_store.get_settings()


@app.put("/api/settings/default_provider")
async def settings_set_default(payload: dict) -> dict:
    provider = (payload or {}).get("provider")
    if not provider:
        raise HTTPException(status_code=400, detail="provider required")
    try:
        return settings_store.set_default_provider(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/settings/providers/{provider}")
async def settings_update_provider(provider: str, payload: dict) -> dict:
    try:
        return settings_store.update_provider(provider, payload or {})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------- folders ----------

@app.get("/api/folders")
def list_folders() -> list[dict]:
    return folder_store.list_folders()


@app.post("/api/folders")
async def create_folder(payload: dict) -> dict:
    name = (payload or {}).get("name", "")
    try:
        return folder_store.create_folder(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.patch("/api/folders/{folder_id}")
async def rename_folder(folder_id: str, payload: dict) -> dict:
    name = (payload or {}).get("name", "")
    try:
        out = folder_store.rename_folder(folder_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if out is None:
        raise HTTPException(status_code=404, detail="folder not found")
    return out


@app.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: str) -> dict:
    result = folder_store.delete_folder(folder_id)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="folder not found")
    if result == "not_empty":
        raise HTTPException(status_code=409, detail="folder is not empty")
    return {"ok": True}


@app.patch("/api/runs/{run_id}/folder")
async def move_run(run_id: str, payload: dict) -> dict:
    if run_path(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    folder_id = (payload or {}).get("folder_id")
    if folder_id is not None and not isinstance(folder_id, str):
        raise HTTPException(status_code=400, detail="folder_id must be a string or null")
    result = folder_store.assign_run(run_id, folder_id or None)
    if result == "unknown_folder":
        raise HTTPException(status_code=404, detail="folder not found")
    return {"ok": True, "run_id": run_id, "folder_id": folder_id or None}


@app.post("/api/runs")
async def create_run(image: UploadFile = File(...)) -> dict:
    if not image.content_type or not image.content_type.startswith(ALLOWED_MIME_PREFIXES):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {image.content_type}")

    ext = Path(image.filename or "").suffix.lower() or mimetypes.guess_extension(image.content_type) or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        ext = ".png"

    run_id, run_dir = new_run()
    dest = run_dir / f"input{ext}"
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    dest.write_bytes(data)

    return {"run_id": run_id, "input": dest.name, "bytes": len(data)}


def _run_pipeline(run_dir: Path, model: str | None, provider: str | None, q: "queue.Queue[dict]") -> None:
    """Worker: drive the agent loop, persist artifacts, push events to the queue.

    The queue holds the same event dicts that the stream yields. A final
    `{"event": "_eof"}` marker tells the SSE coroutine to close the stream.
    """
    image_path = find_input_image(run_dir)
    if image_path is None:
        q.put({"event": "error", "message": "No input image saved for this run."})
        q.put({"event": "_eof"})
        return

    extraction_dump: dict | None = None
    metadata: dict | None = None

    try:
        stream = extract_figure_stream(
            image_path,
            model=model or DEFAULT_MODEL,
            provider=provider or None,
        )
        for ev in stream:
            q.put(ev)
            if ev.get("event") == "complete":
                extraction_dump = ev.get("extraction")
                metadata = ev.get("metadata")
    except Exception as exc:
        q.put({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
        q.put({"event": "_eof"})
        return

    if extraction_dump is None:
        q.put({"event": "_eof"})
        return

    try:
        extraction = FigureExtraction.model_validate(extraction_dump)
    except Exception as exc:
        q.put({"event": "error", "message": f"Schema validation failed: {exc}"})
        q.put({"event": "_eof"})
        return

    (run_dir / "extraction.json").write_text(
        extraction.model_dump_json(indent=2), encoding="utf-8"
    )

    csv_path = run_dir / "reactions.csv"
    try:
        rows = figure_to_rows(extraction, source_image=image_path.name)
        write_csv(rows, csv_path)
    except Exception as exc:
        q.put({"event": "error", "message": f"CSV writing failed: {exc}"})
        q.put({"event": "_eof"})
        return

    q.put(
        {
            "event": "done",
            "extraction": "extraction.json",
            "csv": "reactions.csv",
            "n_reactions": len(extraction.reactions),
            "metadata": metadata,
        }
    )
    q.put({"event": "_eof"})


@app.get("/api/runs/{run_id}/events")
def stream_events(run_id: str, model: str | None = None, provider: str | None = None):
    run_dir = run_path(run_id)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="run not found")

    q: "queue.Queue[dict]" = queue.Queue()
    t = threading.Thread(
        target=_run_pipeline,
        args=(run_dir, model, provider, q),
        daemon=True,
    )
    t.start()

    def gen():
        # Initial comment lets the EventSource open promptly behind some proxies.
        yield ": stream open\n\n"
        while True:
            ev = q.get()
            if ev.get("event") == "_eof":
                return
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/runs/{run_id}/file/{name:path}")
def get_file(run_id: str, name: str):
    run_dir = run_path(run_id)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="run not found")
    target = safe_child(run_dir, name)
    if target is None:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)


@app.post("/api/runs/{run_id}/insight")
def run_insight(run_id: str) -> JSONResponse:
    run_dir = run_path(run_id)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="run not found")

    extraction_file = run_dir / "extraction.json"
    if not extraction_file.exists():
        raise HTTPException(status_code=409, detail="no extraction available yet for this run")

    if not RXN_INSIGHT_VENV.exists():
        return JSONResponse(
            {"ok": False, "error": "Rxn-INSIGHT venv is not configured. See README."},
            status_code=503,
        )

    image_path = find_input_image(run_dir)
    extraction = FigureExtraction.model_validate_json(extraction_file.read_text())
    try:
        analyses = analyze_extraction(
            extraction,
            source_image=image_path.name if image_path else "",
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    rows = extraction_to_insight_rows(
        extraction, analyses,
        source_image=image_path.name if image_path else "",
    )
    (run_dir / "insight.json").write_text(json.dumps(analyses, indent=2), encoding="utf-8")
    write_insight_csv(rows, run_dir / "insight.csv")

    return JSONResponse(
        {
            "ok": True,
            "analyses": analyses,
            "csv": "insight.csv",
        }
    )
