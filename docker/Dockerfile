# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Installs all dependencies and builds wheels
# This stage is heavy — not shipped to production
FROM python:3.11-slim AS builder

WORKDIR /app

# install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: Production ────────────────────────────────────────────────────────
# Copies only built wheels — no build tools, smaller + safer image
FROM python:3.11-slim AS production

WORKDIR /app

# install runtime system dependencies only
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# copy built wheels from builder stage
COPY --from=builder /wheels /wheels

# install from pre-built wheels — fast, no compilation
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# copy application code
COPY . .

# create non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# expose FastAPI port
EXPOSE 8000

# health check — AWS ECS uses this to verify container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# start FastAPI with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]


# ── Stage 3: Streamlit UI ──────────────────────────────────────────────────────
# Separate stage for Streamlit — deployed as separate ECS container
FROM production AS streamlit

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8501/_stcore/health').raise_for_status()"

CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
