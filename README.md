# CJ Studio

Internal tools for Construction Junction: product-photo enhancement (listing
formatter) and a social-media graphic generator.

## Architecture

- **`backend/`** — FastAPI API (Python). Owns all image processing:
  background removal (withoutbg), optional Real-ESRGAN upscale, canvas
  composition, text overlay, and the Figma-derived social templates.
  - `pipeline.py` — listing pipeline (framework-free)
  - `social_engine.py` — social template renderer (framework-free)
  - `main.py` — HTTP endpoints
- **`frontend/`** — Next.js app (TypeScript, App Router). All UI:
  - `/` — Listing workflow: multi-photo upload → batch processing →
    before/after compare, per-photo ratio/text-overlay edits, filmstrip,
    export selection (PNG/ZIP)
  - `/social` — Social Media Generator: upload + title/subtitle → categorized
    template filmstrip (Cover / Text / Secondary / Image; templates the current
    content can't use dim in place) → live preview → download
## Run (development)

```bash
# 1. backend (port 8000)
pip install -r requirements.txt
uvicorn backend.main:app --port 8000 --timeout-keep-alive 75

# 2. frontend (port 3000; /api/* proxies to :8000)
cd frontend && npm install && npm run dev
```

Open http://localhost:3000.

## Deployment

The two halves deploy independently:

- **Backend** — any Python host with ≥4 GB RAM (Hugging Face Spaces, Render,
  Cloud Run…). First request downloads the withoutbg weights from Hugging
  Face. `models/RealESRGAN_x4plus.pth` (64 MB, git-ignored) enables AI
  upscaling; without it the pipeline falls back to LANCZOS.
- **Frontend** — Vercel / any Node host, or `next build` output on a static
  host. Set `BACKEND_URL` so `/api/*` rewrites point at the deployed backend.

## Assets

- `static/social2/` — decoration assets for the social template set
  (white+alpha masks keyed from 4x Figma exports; see
  `scripts/prep_social2_assets.py`)
- `static/social/placeholder_icon.png` — glyph for the neutral preview base
- `fonts/IBMPlexSerif-*.ttf`, `fonts/IBMPlexSans-Medium.ttf` — template
  typography (Plex Sans stands in for Helvetica Neue in the Secondary-1
  byline)
