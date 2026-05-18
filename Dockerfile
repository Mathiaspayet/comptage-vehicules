# ============================================================
# Dockerfile — Comptage de véhicules
# Architecture cible : linux/amd64 (Intel Celeron du DS218+)
# La construction se fait sur GitHub Actions, PAS sur le NAS.
# ============================================================

FROM python:3.11-slim

# --- Dépendances système pour OpenCV (headless) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Installation des dépendances Python ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Version (injectée par GitHub Actions) ---
ARG GIT_SHA=dev
ARG BUILD_DATE=unknown
RUN echo "{\"sha\": \"${GIT_SHA}\", \"built_at\": \"${BUILD_DATE}\"}" > /app/version.json

# --- Code source ---
COPY src/ ./src/

# --- Dossier de données (sera monté en volume sur le NAS) ---
RUN mkdir -p /app/data/models /app/data/logs

# --- Variables d'environnement par défaut ---
ENV CONFIG_PATH=/app/data/config.yaml \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- Ports exposés ---
# 8080 : tableau de bord web
# 8081 : outil de calibration
EXPOSE 8080 8081

# --- Point d'entrée ---
CMD ["python", "-m", "src.main"]
