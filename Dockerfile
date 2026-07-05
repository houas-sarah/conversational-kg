FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN mkdir -p /app/data && chmod -R 777 /app/data

ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV HOST=0.0.0.0

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.main:app --host ${HOST} --port ${PORT}"]
