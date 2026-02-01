# ---- Base image ----
FROM python:3.11-slim

# ---- System deps (for soundfile/libsndfile + common audio tooling) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- Python env hardening ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- Install Python deps first (better Docker caching) ----
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Copy the rest of the repo ----
COPY . /app

# ---- Streamlit settings for containers ----
# Render/Railway provide $PORT at runtime. Locally we can set it.
ENV PORT=8501
EXPOSE 8501

CMD ["bash", "-lc", "streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]