# ============================================================
# Dockerfile — RekomendasiSahamV2
# Base: python:3.9-slim  |  Server: gunicorn  |  Port: 8000
# ============================================================

FROM python:3.9-slim

# ── System dependencies ──────────────────────────────────────
# libgomp1  : diperlukan oleh XGBoost (OpenMP runtime)
# libhdf5-dev tidak perlu karena model .h5 di-load via tensorflow langsung
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Buat user non-root untuk keamanan ─────────────────────────
RUN groupadd -r appuser && useradd -r -g appuser appuser

# ── Working directory ──────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────
# Copy requirements dulu agar layer ini di-cache selama kode belum berubah
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy seluruh source code ──────────────────────────────────
COPY . .

# ── Pastikan direktori data & models bisa dibaca ──────────────
RUN chown -R appuser:appuser /app

# ── Ganti ke user non-root ─────────────────────────────────────
USER appuser

# ── Expose port 8000 (sesuai fly.toml) ────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Jalankan dengan gunicorn ──────────────────────────────────
# --workers 2    : 2 worker process (sesuaikan dengan RAM Fly.io; naik ke 4 jika 1 GB+)
# --threads 2    : 2 thread per worker untuk I/O-bound (news fetch, yfinance)
# --timeout 120  : timeout lebih panjang untuk request NLP yang berat
# --bind         : bind ke semua interface pada port 8000
CMD ["gunicorn", "app:app", \
     "--workers", "2", \
     "--threads", "2", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
