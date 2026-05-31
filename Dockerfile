FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer — changes rarely) ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code only (data/models/mlruns excluded via .dockerignore) ─────
COPY configs/ configs/
COPY src/     src/

# Runtime dirs for local docker runs (Cloud Run uses /tmp/ via GCS download)
RUN mkdir -p data/raw data/processed data/predictions models mlruns logs

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "src/pipeline.py"]
