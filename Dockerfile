# CJ Studio — FastAPI backend image.
# Deployable to Hugging Face Spaces (Docker SDK, default port 7860), Render,
# or Cloud Run (both set $PORT). The Next.js frontend deploys separately on
# Vercel and proxies /api/* here via its BACKEND_URL.
FROM python:3.11-slim

# System libraries:
#   opencv-python-headless → libgl1, libglib2.0-0
#   Pillow                 → libjpeg62-turbo, libpng16-16, libwebp7
#   text-overlay fonts     → fonts-dejavu-core, fonts-liberation
#   healthcheck / TLS      → curl, ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libjpeg62-turbo \
        libpng16-16 \
        libwebp7 \
        fonts-dejavu-core \
        fonts-liberation \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Hugging Face Spaces runs the container as uid 1000; give that user a writable
# home so the withoutbg model cache and any temp files land somewhere valid.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    CJ_UPSCALE=0
WORKDIR /home/user/app

# App code + the runtime assets the pipeline reads (see .dockerignore for what
# stays out — the frontend and the legacy Streamlit app are not needed here).
COPY --chown=user backend/ backend/
# Whole static/ tree (~11MB): social2 template masks, product_bg.png backdrop,
# cover scene backgrounds, placeholder icon. Copying selectively has already
# bitten us once (social2 + product_bg missing → Social 500s, wrong backdrop).
COPY --chown=user static/ static/
COPY --chown=user fonts/ fonts/

USER user

# NOTE: the legacy withoutbg background-removal model is intentionally NOT
# baked in — this deployment runs the OpenAI pipeline (New Listing / Social)
# only, which keeps the image small and the RAM footprint low. To re-enable
# legacy, add withoutbg + onnxruntime back to requirements-backend.txt and
# restore the snapshot_download RUN step here.

EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --timeout-keep-alive 75"]
