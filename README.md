# CJ Studio

Internal tools for Construction Junction: product-photo enhancement (listing
formatter) and a social-media graphic generator.

## Architecture

- **`backend/`** — FastAPI API (Python). Owns all image processing:
  - `pipeline.py` — listing pipeline: gpt-image-2 redraws the product on a
    white background with a studio shadow, multiply-blended onto the CJ
    backdrop; Cover mode composes the product into a per-category scene. The
    pre-AI local pipeline (withoutbg background removal, optional Real-ESRGAN
    upscale, text overlay) lives on behind `/legacy`.
  - `openai_engine.py` — all OpenAI calls: white-plate generation and cover
    scene composition (gpt-image-2), cover category classification (gpt-5.4).
    Cover mode classifies the product into one of five categories (Furniture,
    Appliances & Fixtures, Building Materials/Outdoor, Lighting, Specialty),
    each with its own scene backdrop and prompt. Models and parameters are
    overridable via `CJ_OPENAI_*` env vars (see `.env.example`).
  - `social_engine.py` — social template renderer (framework-free)
  - `main.py` — HTTP endpoints
- **`frontend/`** — Next.js app (TypeScript, App Router). All UI:
  - `/` — New Listing workflow: multi-photo upload → batch processing (3 in
    parallel, ~15–50 s per photo) → before/after compare, per-photo
    ratio/dimension edits, optional cover generation for featured items,
    filmstrip, export selection (PNG/ZIP)
  - `/social` — Social Media Generator: upload + title/subtitle → categorized
    template filmstrip (Cover / Text / Secondary / Image; templates the current
    content can't use dim in place) → live preview → download
  - `/library` — the last 50 exported photos, saved automatically on every
    export (deduped, survives restarts)
  - `/legacy` — the original local pipeline; runs on the host machine and
    uses no AI credits

## Run (development)

```bash
# 0. OpenAI key (required for / and cover generation; /legacy runs without it)
cp .env.example .env   # then fill in OPENAI_API_KEY

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
  Cloud Run…). Needs `OPENAI_API_KEY` in the environment. The legacy path
  downloads the withoutbg weights from Hugging Face on first request.
  `models/RealESRGAN_x4plus.pth` (64 MB, git-ignored) enables AI upscaling;
  without it the pipeline falls back to LANCZOS.
- **Frontend** — Vercel / any Node host, or `next build` output on a static
  host. Set `BACKEND_URL` so `/api/*` rewrites point at the deployed backend.

## Assets

- `static/cover/` — the five Cover-mode scene backdrops, one per category
  (the Building Materials/Outdoor file ships as `Outdoor Background.png` —
  "/" can't appear in a filename)
- `static/social2/` — decoration assets for the social template set
  (white+alpha masks keyed from 4x Figma exports; see
  `scripts/prep_social2_assets.py`)
- `static/social/placeholder_icon.png` — glyph for the neutral preview base
- `fonts/IBMPlexSerif-*.ttf`, `fonts/IBMPlexSans-Medium.ttf` — template
  typography (Plex Sans stands in for Helvetica Neue in the Secondary-1
  byline)
