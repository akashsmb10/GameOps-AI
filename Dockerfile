# GameOps AI - one image for the API and the dashboard (CPU only, no GPU or credentials needed)
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Build the demo pipeline into the image so containers start ready (synthetic data only).
RUN python scripts/build_pipeline.py && python -m src.evaluation.rag_eval && python -m src.evaluation.agent_eval > /dev/null
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
