FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY models ./models
COPY data/test ./data/test
COPY data/insights.json ./data/insights.json
COPY data/train ./data/train

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s CMD curl -fs http://localhost:${PORT}/health || exit 1
# One worker per container: scale by adding containers, which is what the load test measures.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
