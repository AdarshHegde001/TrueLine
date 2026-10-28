FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        build-essential \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/package.json backend/package-lock.json ./backend/
RUN cd backend && npm ci --omit=dev

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY rag/requirements.txt /tmp/rag-requirements.txt
COPY data_ingestion/requirements.txt /tmp/ingestion-requirements.txt
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/rag-requirements.txt -r /tmp/ingestion-requirements.txt

COPY backend ./backend
COPY rag ./rag
COPY data_ingestion ./data_ingestion

EXPOSE 3001
CMD ["npm", "--prefix", "backend", "start"]
