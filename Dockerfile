FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY config ./config

RUN mkdir -p /app/data/raw/pdf \
             /app/data/processed/parsed \
             /app/data/processed/claims \
             /app/data/processed/classifications \
             /app/data/processed/dimensional_reports \
             /app/data/processed/sympy_reports \
             /app/data/processed/novelty_reports \
             /app/data/processed/structure_reports \
             /app/data/processed/domain_reports \
             /app/data/processed/reproducibility_reports \
             /app/data/processed/llm_detection_reports \
             /app/data/processed/math_reports \
             /app/data/processed/referee_reports \
             /app/data/processed/referee_sections \
             /app/data/processed/triage_scores \
             /app/data/processed/evaluations

ENV PYTHONPATH=/app/src \
    PEERLENS_DATA_ROOT=/app/data \
    PORT=8080 \
    HOST=0.0.0.0

EXPOSE 8080

CMD ["python", "scripts/run_webapp.py"]
