# Backend image (FastAPI + Celery worker + Playwright)
FROM python:3.12-slim

# Force apt to use HTTPS to avoid ISP login page interception
RUN sed -i 's|^URIs: http://deb.debian.org|URIs: https://deb.debian.org|' /etc/apt/sources.list.d/debian.sources

# Install system dependencies for Playwright (Chromium)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpango-1.0-0 libpangoft2-1.0-0 libcairo2 \
        libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info \
        fonts-dejavu curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install Playwright Chromium
RUN playwright install chromium

# Copy application code
COPY backend/ ./

# Create storage directories
RUN mkdir -p /app/storage/uploads /app/storage/reports /app/storage/figures /app/storage/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]