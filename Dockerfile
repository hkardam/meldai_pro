FROM python:3.12-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface

# Install system dependencies (C compilers for psycopg, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt pyproject.toml ./

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 --retries 10 -r requirements.txt

# Create cache and data directories
RUN mkdir -p /app/cache/huggingface /app/data /app/logs

# Copy application source code and entrypoint
COPY src/ ./src/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Install the local package in editable mode so changes in ./src reflect immediately
RUN pip install --no-cache-dir -e .

# Expose debugpy port (5678) and future web app port (8000)
EXPOSE 5678 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "meldai.main", "run-demo"]
