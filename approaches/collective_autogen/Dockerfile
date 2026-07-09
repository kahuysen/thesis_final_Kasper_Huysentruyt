# Multi-stage build:
#   1. frontend-build — compiles the Vite/React SPA to webapp/frontend/dist/
#   2. runtime         — Python backend + bundled frontend assets
#
# v1 ships with only the main Python venv. The MolDetect (rxnim) and MolNexTR
# venvs are not bundled — those need GPU + extra checkpoint volumes, and
# bringing them in doubles the image size. Without them, main.py runs in its
# default mode (no --use-molnextr): the molecular_recognition agent uses the
# LLM's vision capability rather than calling MolDetect/MolNexTR. This is
# the same default as `python main.py image.png` from the host.

# ----------------------------------------------------------------------------
# Stage 1 — build the React frontend
# ----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-build
WORKDIR /app/webapp/frontend

# Install deps separately so this layer is cached when only source changes.
COPY webapp/frontend/package.json webapp/frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY webapp/frontend/ ./
RUN npm run build

# ----------------------------------------------------------------------------
# Stage 2 — Python runtime
# ----------------------------------------------------------------------------
# Switch base to nvidia/cuda:12.x-runtime-ubuntu22.04 when bundling MolDetect.
FROM python:3.12-slim AS runtime

# poppler-utils provides pdftoppm/pdftocairo for pdf2image. tesseract is
# needed by tools/ocr.py at run time. libgl1 + libglib2.0-0 are runtime
# deps for opencv (used inside Florence-2 / MolDetect downstream).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      poppler-utils tesseract-ocr libgl1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY webapp/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Source. We exclude runs/ uploads/ via .dockerignore so the image stays slim.
COPY . /app

# Frontend dist from stage 1.
COPY --from=frontend-build /app/webapp/frontend/dist /app/webapp/frontend/dist

# Make sure the runtime dirs the backend writes into exist as volumes.
RUN mkdir -p /app/runs /app/webapp/uploads

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "webapp.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
