FROM python:3.11-slim

# Set environment variables for minimal memory footprint (512MB RAM containers)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies (CPU-only PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure data folders exist with write permissions
RUN mkdir -p data/papers data/processed data/chunks data/chroma_db data/users data/evaluations data/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/api/health || exit 1

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
