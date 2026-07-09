# ReactionMiner — Web App

A FastAPI wrapper around the single-SDK chemistry vision-LLM pipeline. Upload a chemistry figure, watch the agent's `validate_smiles` and `submit_extraction` tool calls stream live, and download the resulting reaction cards, structured JSON, and flat CSV.

## Layout

```
pipeline/              vision-agent pipeline (extractor, renderer, flatten, …)
prompts/               system prompts loaded by the extractor
subprocess_drivers/    Rxn-INSIGHT runner (optional enrichment)
app/
  main.py              FastAPI routes (upload, SSE stream, file serve, health, insight)
  runs.py              per-run output dir helpers
  static/              index.html + style.css + script.js (no build step)
runs/                  generated per-session output (gitignored)
```

## Setup

```bash
cd "5 App"
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in your provider creds (see below)
```

### Provider configuration (`.env`)

Pick one (or several — the UI shows a picker over every provider whose keys are set):

- **Direct Anthropic**: `ANTHROPIC_API_KEY=sk-ant-...`
- **Azure-hosted Claude**: `LLM_PROVIDER=azure`, `AZURE_ANTHROPIC_ENDPOINT`, `AZURE_ANTHROPIC_API_KEY`, `AZURE_DEPLOYMENT_NAME`
- **Google Gemini**: `LLM_PROVIDER=gemini`, `GEMINI_API_KEY` (`pip install google-genai`)
- **OpenRouter**: `OPENROUTER_API_KEY` (`pip install openai`)

## Run

```bash
.venv/bin/uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
```

## Optional: Rxn-INSIGHT enrichment

Rxn-INSIGHT pins numpy 1.x and runs in its own venv. When `.venv-rxn-insight/` exists, the UI shows an "Analyze with Rxn-INSIGHT" button after extraction.

```bash
python3.12 -m venv .venv-rxn-insight
.venv-rxn-insight/bin/pip install rxn-insight 'scipy>=1.11,<1.15' joblib
```

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/`                            | static UI |
| `GET`  | `/api/health`                  | provider, default model, capability flags |
| `POST` | `/api/runs`                    | upload one image, returns `{run_id}` |
| `GET`  | `/api/runs/{id}/events`        | Server-Sent Events stream of agent steps |
| `GET`  | `/api/runs/{id}/file/{name}`   | serve any artifact under `runs/<id>/` |
| `POST` | `/api/runs/{id}/insight`       | run Rxn-INSIGHT (503 if venv missing) |

## Notes

- Server-Sent Events feed `step_start`, `step_done`, `complete`, `done`, and `error` events to the page.
- Cards are rendered with RDKit + matplotlib on the server (`pipeline.renderer.render_figure`).
- All artifacts live under `runs/<uuid>/` and are gitignored.
