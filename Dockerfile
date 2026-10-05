# ============================================================
# Stage 1: Builder — full image with Rust & build tools
#           compiles thinc, spaCy, medspaCy and all C extensions
# ============================================================
FROM python:3.12-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System build tools + Rust (required by thinc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt pyproject.toml ./

# --prefer-binary: always use pre-built wheels (spaCy 3.8 + thinc 8.3 have Python 3.12 wheels).
# Download en_core_web_sm and en_core_web_md models.
# scispacy is not used — HPO codes are resolved via HPOService (pronto, offline).
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary --default-timeout=120 --retries 10 -r requirements.txt && \
    python -m spacy download en_core_web_md && \
    python -c "import spacy; import medspacy; nlp = spacy.load('en_core_web_md'); nlp.add_pipe('medspacy_context'); doc = nlp('Patient has no fever'); print('medspaCy pipeline OK:', nlp.pipe_names)"


# ============================================================
# Stage 2: Runtime — slim image, copy compiled site-packages
# ============================================================
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface

# Minimal runtime system deps (libpq for psycopg, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all compiled Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create cache and data directories
RUN mkdir -p /app/cache/huggingface /app/data /app/logs

# Copy application source code and entrypoint
COPY src/ ./src/
COPY docker/entrypoint.sh /entrypoint.sh
COPY scripts/ ./scripts/
RUN chmod +x /entrypoint.sh

# Copy pyproject.toml required for editable package installation
COPY pyproject.toml ./
RUN pip install --no-cache-dir --no-deps -e .

# Expose debugpy port (5678) and FastAPI port (8000)
EXPOSE 5678 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "meldai.main", "run-demo"]

