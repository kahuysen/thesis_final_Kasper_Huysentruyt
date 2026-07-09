"""Batch orchestrator: walk a corpus of figures, run the agent on each,
write per-figure JSON + cards, and append rows to a master CSV.

Concurrency is bounded with a thread pool because the underlying SDK call
is blocking I/O. For very large corpora consider switching to the async
client + asyncio.Semaphore.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from anthropic import Anthropic

from .config import build_client, build_client_for, current_provider, provider_for_model
from .extractor import DEFAULT_MODEL, extract_figure
from .flatten import figure_to_rows, write_csv
from .renderer import render_figure
from .schema import FigureExtraction


def _write_meta_sidecar(out_dir: Path, stem: str, meta: dict, elapsed_s: float) -> None:
    """Write `<stem>.meta.json` with the same shape scripts/_meta_helpers.py
    produces, so scripts/summarize_costs.py can aggregate this directory."""
    sidecar = {
        "stem": stem,
        "backend": provider_for_model(meta.get("model")) if meta.get("model") else None,
        "model": meta.get("model"),
        "steps": meta.get("steps"),
        "tool_calls": meta.get("tool_calls"),
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "elapsed_s": round(float(elapsed_s), 2),
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(sidecar, indent=2))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@dataclass
class FigureResult:
    image: Path
    json_path: Path | None = None
    cards_dir: Path | None = None
    rows: list[dict] | None = None
    error: str | None = None
    metadata: dict | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _list_images(corpus: Path) -> list[Path]:
    return sorted(p for p in corpus.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _process_one(
    image: Path,
    out_root: Path,
    *,
    model: str,
    skip_existing: bool,
    client: Anthropic,
) -> FigureResult:
    stem = image.stem
    json_path = out_root / f"{stem}.json"
    cards_dir = out_root / f"{stem}_cards"

    if skip_existing and json_path.exists():
        try:
            fx = FigureExtraction.model_validate_json(json_path.read_text())
            return FigureResult(
                image=image,
                json_path=json_path,
                cards_dir=cards_dir if cards_dir.exists() else None,
                rows=figure_to_rows(fx, source_image=image.name),
                metadata={"skipped": True},
            )
        except Exception as e:
            return FigureResult(image=image, error=f"existing JSON unreadable: {e}")

    t0 = time.perf_counter()
    try:
        fx, meta = extract_figure(image, model=model, client=client)
    except Exception as e:
        return FigureResult(image=image, error=f"extraction failed: {e}")
    elapsed_s = time.perf_counter() - t0

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(fx.model_dump_json(indent=2))
    try:
        _write_meta_sidecar(out_root, stem, meta, elapsed_s)
    except Exception as e:
        print(f"  WARN  meta sidecar write failed for {stem}: {e}")

    try:
        render_figure(fx, cards_dir)
    except Exception as e:
        return FigureResult(
            image=image, json_path=json_path, error=f"rendering failed: {e}", metadata=meta
        )

    return FigureResult(
        image=image,
        json_path=json_path,
        cards_dir=cards_dir,
        rows=figure_to_rows(fx, source_image=image.name),
        metadata=meta,
    )


def run_batch(
    corpus: str | Path,
    out_root: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 4,
    skip_existing: bool = True,
    csv_name: str = "reactions.csv",
    progress: bool = True,
) -> list[FigureResult]:
    corpus = Path(corpus)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    images = _list_images(corpus)
    if not images:
        print(f"No images found under {corpus}")
        return []

    # Build the client for whichever provider services the requested model,
    # not blindly the default LLM_PROVIDER — otherwise --model that crosses
    # providers (e.g. anthropic/claude-opus-4-7 via OpenRouter while LLM_PROVIDER=azure)
    # would reuse the wrong client and crash inside the dispatcher.
    resolved_provider = provider_for_model(model) or current_provider()
    client = build_client_for(resolved_provider)
    results: list[FigureResult] = []

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(_process_one, img, out_root,
                      model=model, skip_existing=skip_existing, client=client): img
            for img in images
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            results.append(res)
            if progress:
                tag = "OK " if res.ok else "ERR"
                extra = ""
                if res.metadata and res.metadata.get("skipped"):
                    tag = "SKIP"
                elif res.metadata:
                    extra = f"  ({res.metadata.get('steps','?')} steps, " \
                            f"{res.metadata.get('tool_calls','?')} tool calls, " \
                            f"in={res.metadata.get('input_tokens','?')}, " \
                            f"out={res.metadata.get('output_tokens','?')})"
                msg = res.error or extra
                print(f"[{i}/{len(images)}] {tag}  {res.image.name}  {msg}")

    all_rows: list[dict] = []
    for r in results:
        if r.rows:
            all_rows.extend(r.rows)
    if all_rows:
        csv_path = write_csv(all_rows, out_root / csv_name)
        if progress:
            print(f"Wrote {len(all_rows)} rows → {csv_path}")

    return results
