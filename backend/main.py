"""CJ Studio API — FastAPI backend for the Next.js frontend.

Listing workflow
    POST   /api/photos                     multipart file → process, store, return id
    GET    /api/photos/{id}/image          composed result (ratio/text/caption/w params)
    GET    /api/photos/{id}/thumb          small JPEG of the enhanced result
    DELETE /api/photos/{id}
    POST   /api/export                     selected photos → single PNG or ZIP

Social workflow
    POST   /api/social/upload              raw image → id
    GET    /api/testing/render             styled PNG for an uploaded/placeholder base

State is in-memory (single-process deployment): photos evicted after
PHOTO_TTL seconds of inactivity.
"""
import hashlib
import io
import json
import logging
import os
import threading
import time
import uuid
import zipfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel

from . import pipeline, social_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cj_api")

app = FastAPI(title="CJ Studio API")

# Dev: Next.js runs on :3000 and proxies /api/* here, so CORS is belt-and-
# braces for direct calls (e.g. curl, other ports).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PHOTO_TTL = 60 * 60 * 6      # 6h — internal tool, generous
_photos: dict = {}           # id → record
_social_imgs: dict = {}      # id → {bytes, touched}
_testing2: dict = {}         # id → {name, plate, original, compose_cache, touched}

# Cap concurrent heavy Pillow renders. Sync endpoints run in a ~40-thread
# pool, so e.g. the Social filmstrip's 17 template thumbnails all render at
# once — each briefly holds a decoded photo plus canvases, and together the
# spike OOM-restarts a small instance (Render free = 512MB), wiping every
# in-memory session. Three at a time bounds the spike; the rest of the
# requests just wait a beat.
_RENDER_SEM = threading.BoundedSemaphore(
    int(os.environ.get("CJ_RENDER_CONCURRENCY", "3")))


def _evict_stale() -> None:
    now = time.time()
    for store in (_photos, _social_imgs, _testing2):
        for k in [k for k, v in store.items() if now - v["touched"] > PHOTO_TTL]:
            store.pop(k, None)


def _get_photo(photo_id: str) -> dict:
    rec = _photos.get(photo_id)
    if rec is None:
        raise HTTPException(404, "photo not found")
    rec["touched"] = time.time()
    return rec


# ── Listing: process / compose / export ────────────────────────────────────────

@app.post("/api/photos")
async def create_photo(file: UploadFile = File(...)):
    _evict_stale()
    raw = await file.read()
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    except Exception:
        raise HTTPException(400, f"{file.filename}: could not open image")

    # Off the event loop: the local pipeline blocks for seconds, which would
    # otherwise stall every other request (thumbs, images) while it runs.
    result, warn, mask, rgba, orig_layer = await run_in_threadpool(
        pipeline.make_listing, img)
    if warn:
        return JSONResponse({"id": None, "name": file.filename, "warn": warn},
                            status_code=422)

    photo_id = uuid.uuid4().hex[:12]
    _photos[photo_id] = {
        "name": file.filename,
        "original": img,
        "result": result,
        "subject_mask": mask,
        "subject_rgba": rgba,
        "orig_layer": orig_layer,
        "compose_cache": {},
        "touched": time.time(),
    }
    return {"id": photo_id, "name": file.filename, "warn": None}


def _composed_png(rec: dict, ratio: str, text_mode: bool, caption: str,
                  kind: str, width: int | None, orig_bg: bool = False) -> bytes:
    key = (kind, ratio, text_mode, caption, width, orig_bg)
    cache = rec["compose_cache"]
    if key in cache:
        return cache[key]

    if kind == "before":
        img = pipeline.fit_to_ratio(rec["original"], ratio)
    else:
        img = pipeline.compose(rec["result"], rec["subject_rgba"],
                               ratio, text_mode, caption,
                               rec.get("orig_layer") if orig_bg else None)
    if width and width < img.width:
        img = img.resize((width, round(img.height * width / img.width)),
                         Image.LANCZOS)
    png = pipeline.to_png_bytes(img)
    if len(cache) > 16:
        cache.clear()
    cache[key] = png
    return png


@app.get("/api/photos/{photo_id}/image")
def photo_image(photo_id: str, ratio: str = "1:1", text: int = 0,
                caption: str = "", kind: str = "after", w: int | None = None,
                origbg: int = 0):
    rec = _get_photo(photo_id)
    png = _composed_png(rec, ratio, bool(text), caption, kind, w, bool(origbg))
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/photos/{photo_id}/thumb")
def photo_thumb(photo_id: str, s: int = 160):
    rec = _get_photo(photo_id)
    thumb = rec["result"].copy()
    thumb.thumbnail((s, s))
    return Response(pipeline.to_jpeg_bytes(thumb, quality=82),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/api/photos/{photo_id}")
def delete_photo(photo_id: str):
    _photos.pop(photo_id, None)
    return {"ok": True}


class DimPoint(BaseModel):
    x: float
    y: float


class DimensionSpec(BaseModel):
    start: DimPoint
    end: DimPoint
    value: str = ""


class ExportItem(BaseModel):
    id: str
    ratio: str = "1:1"
    text_mode: bool = False
    caption: str = ""
    orig_bg: bool = False
    dimensions: list[DimensionSpec] = []


class ExportRequest(BaseModel):
    items: list[ExportItem]


@app.post("/api/export")
def export_photos(req: ExportRequest):
    if not req.items:
        raise HTTPException(400, "nothing selected")

    def _png(item: ExportItem) -> tuple[str, bytes]:
        rec = _get_photo(item.id)
        png = _composed_png(rec, item.ratio, item.text_mode, item.caption,
                            "after", None, item.orig_bg)
        if item.dimensions:
            img = pipeline.draw_dimensions(
                Image.open(io.BytesIO(png)),
                [d.model_dump() for d in item.dimensions])
            png = pipeline.to_png_bytes(img)
        stem = (rec["name"].rsplit(".", 1)[0] or "photo")
        return f"cj_{stem}_{item.ratio.replace(':', 'x')}.png", png

    if len(req.items) == 1:
        name, png = _png(req.items[0])
        return Response(png, media_type="image/png", headers={
            "Content-Disposition": f'attachment; filename="{name}"'})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        used = set()
        for item in req.items:
            name, png = _png(item)
            while name in used:                      # dedupe repeated stems
                name = name.replace(".png", "_1.png")
            used.add(name)
            zf.writestr(name, png)
    return Response(buf.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": 'attachment; filename="cj_photos.zip"'})


# ── Testing 2: OpenAI white-plate + multiply composition ────────────────────────

@app.post("/api/testing2")
async def create_testing2(file: UploadFile = File(...)):
    _evict_stale()
    raw = await file.read()
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        # Bound the retained original: the gpt-image input downscales to
        # 2048px anyway and the "before" preview tops out at 1400px, but a
        # 12MP original held as RGB is ~36MB of resident RAM per photo —
        # a handful of uploads alone can OOM a 512MB instance.
        if max(img.size) > 2048:
            img.thumbnail((2048, 2048), Image.LANCZOS)
    except Exception:
        raise HTTPException(400, f"{file.filename}: could not open image")

    # Blocking call off the event loop, so concurrent uploads actually run in
    # parallel (the frontend batches up to 3 at once) and image/thumb requests
    # keep serving while a generation is in flight.
    plate, warn = await run_in_threadpool(pipeline.make_listing_openai, img)
    if plate is None:
        return JSONResponse({"id": None, "name": file.filename, "warn": warn},
                            status_code=422)

    tid = uuid.uuid4().hex[:12]
    _testing2[tid] = {
        "name": file.filename,
        "plate": plate,
        "original": img.convert("RGB"),
        "compose_cache": {},
        "touched": time.time(),
    }
    return {"id": tid, "name": file.filename, "warn": None}


def _get_testing2(tid: str) -> dict:
    rec = _testing2.get(tid)
    if rec is None:
        raise HTTPException(404, "not found")
    rec["touched"] = time.time()
    return rec


def _testing2_png(rec: dict, ratio: str, kind: str, width: int | None) -> bytes:
    key = (kind, ratio, width)
    cache = rec["compose_cache"]
    if key in cache:
        return cache[key]

    with _RENDER_SEM:
        if kind == "before":
            img = pipeline.fit_to_ratio(rec["original"], ratio)
        elif kind == "cover":
            if rec.get("cover") is None:
                raise HTTPException(404, "cover not generated")
            img = pipeline.cover_to_ratio(rec["cover"], ratio)
        else:
            img = pipeline.compose_testing2(rec["plate"], ratio)
        if width and width < img.width:
            img = img.resize((width, round(img.height * width / img.width)),
                             Image.LANCZOS)
        # JPEG for on-screen previews: a 1400px stage PNG is ~2MB vs ~250KB as
        # JPEG-85 — on mobile connections that difference IS the loading time.
        # Downloads keep full-quality PNG via the separate /export path.
        jpg = pipeline.to_jpeg_bytes(img, quality=85)
    if len(cache) > 16:
        cache.clear()
    cache[key] = jpg
    return jpg


@app.get("/api/testing2/{tid}/image")
def testing2_image(tid: str, ratio: str = "1:1", kind: str = "after",
                   w: int | None = None):
    rec = _get_testing2(tid)
    jpg = _testing2_png(rec, ratio, kind, w)
    # A given (id, kind, ratio, w) never changes once it exists (the plate is
    # fixed at upload; covers get their own kind and 404 until generated), so
    # let the browser cache it — the frontend prefetches the other ratios and
    # relies on these being cache hits for instant ratio switching. `private`
    # keeps shared proxies out of it. no-store here made every ratio switch a
    # full re-download and the prefetch pure wasted bandwidth.
    return Response(jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/testing2/{tid}/cover")
def testing2_cover(tid: str):
    """Generate (and cache) the cover scene for one photo.

    Idempotent: repeat calls return the cached result without re-billing.
    """
    rec = _get_testing2(tid)
    if rec.get("cover") is None:
        cover, category, warn = pipeline.make_cover(rec["original"])
        if cover is None:
            return JSONResponse({"ok": False, "category": category, "warn": warn},
                                status_code=422)
        rec["cover"] = cover
        rec["cover_category"] = category
    return {"ok": True, "category": rec.get("cover_category")}


@app.delete("/api/testing2/{tid}/cover")
def testing2_cover_delete(tid: str):
    """Drop the cached cover so Remove Cover is genuinely destructive.

    The UI warns "Removing a cover cannot be undone" — without this, the POST
    route's cache would hand the same image straight back. After this,
    regenerating is a fresh billed run with a different result.
    """
    rec = _get_testing2(tid)
    rec.pop("cover", None)
    rec.pop("cover_category", None)
    return {"ok": True}


@app.get("/api/testing2/{tid}/thumb")
def testing2_thumb(tid: str, s: int = 160, kind: str = "after"):
    rec = _get_testing2(tid)
    if kind == "cover":
        if rec.get("cover") is None:
            raise HTTPException(404, "cover not generated")
        thumb = pipeline.cover_to_ratio(rec["cover"], "1:1")
    else:
        thumb = pipeline.compose_testing2(rec["plate"], "1:1")
    thumb.thumbnail((s, s))
    return Response(pipeline.to_jpeg_bytes(thumb, quality=82),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/api/testing2/{tid}")
def delete_testing2(tid: str):
    _testing2.pop(tid, None)
    return {"ok": True}


class Testing2ExportItem(BaseModel):
    id: str
    ratio: str = "1:1"
    kind: str = "after"                  # "after" (white-plate) | "cover" (scene)
    dimensions: list[DimensionSpec] = []


class Testing2ExportRequest(BaseModel):
    items: list[Testing2ExportItem]


@app.post("/api/testing2/export")
def testing2_export(req: Testing2ExportRequest):
    if not req.items:
        raise HTTPException(400, "nothing selected")

    def _png(item: Testing2ExportItem) -> tuple[str, bytes]:
        rec = _get_testing2(item.id)
        if item.kind == "cover":
            if rec.get("cover") is None:
                raise HTTPException(404, "cover not generated")
            img = pipeline.cover_to_ratio(rec["cover"], item.ratio)
        else:
            img = pipeline.compose_testing2(rec["plate"], item.ratio)
        if item.dimensions:
            img = pipeline.draw_dimensions(
                img, [d.model_dump() for d in item.dimensions])
        stem = (rec["name"].rsplit(".", 1)[0] or "photo")
        suffix = "_cover" if item.kind == "cover" else ""
        fname = f"cj_{stem}{suffix}_{item.ratio.replace(':', 'x')}.png"
        png = pipeline.to_png_bytes(img)
        # Every export lands in the Library automatically (50 most recent).
        _library_add(fname, item.kind, item.ratio, png)
        return fname, png

    if len(req.items) == 1:
        name, png = _png(req.items[0])
        return Response(png, media_type="image/png", headers={
            "Content-Disposition": f'attachment; filename="{name}"'})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        used = set()
        for item in req.items:
            name, png = _png(item)
            while name in used:
                name = name.replace(".png", "_1.png")
            used.add(name)
            zf.writestr(name, png)
    return Response(buf.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": 'attachment; filename="cj_photos.zip"'})


# ── Library: exports auto-saved to disk (survives restarts) ────────────────────

_LIBRARY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "storage", "library")
_LIBRARY_MAX = 50
_library_lock = threading.Lock()


def _library_index_path() -> str:
    return os.path.join(_LIBRARY_DIR, "index.json")


def _library_load() -> list:
    """Read the index fresh from disk (≤50 small records — cheap and simple)."""
    try:
        with open(_library_index_path()) as f:
            return json.load(f)
    except Exception:
        return []


def _library_write(entries: list) -> None:
    os.makedirs(_LIBRARY_DIR, exist_ok=True)
    tmp = _library_index_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entries, f)
    os.replace(tmp, _library_index_path())


def _library_add(name: str, kind: str, ratio: str, png: bytes) -> None:
    """Save one exported PNG into the library (newest first).

    Re-exporting identical content replaces the older entry instead of
    duplicating it; beyond _LIBRARY_MAX the oldest entries (and their files)
    are evicted. Never raises — a library hiccup must not break an export.
    """
    try:
        digest = hashlib.sha256(png).hexdigest()[:16]
        img = Image.open(io.BytesIO(png))
        with _library_lock:
            os.makedirs(_LIBRARY_DIR, exist_ok=True)
            entries = _library_load()
            for stale in [e for e in entries if e.get("hash") == digest]:
                entries.remove(stale)
                try:
                    os.remove(os.path.join(_LIBRARY_DIR, f"{stale['id']}.png"))
                except OSError:
                    pass
            lid = uuid.uuid4().hex[:12]
            with open(os.path.join(_LIBRARY_DIR, f"{lid}.png"), "wb") as f:
                f.write(png)
            entries.insert(0, {
                "id": lid, "name": name, "kind": kind, "ratio": ratio,
                "w": img.width, "h": img.height, "hash": digest,
                "created": time.time(),
            })
            for evicted in entries[_LIBRARY_MAX:]:
                try:
                    os.remove(os.path.join(_LIBRARY_DIR, f"{evicted['id']}.png"))
                except OSError:
                    pass
            _library_write(entries[:_LIBRARY_MAX])
    except Exception as exc:
        log.warning("library_add failed (%s) — export continues", exc)


def _library_entry(lid: str) -> dict:
    for e in _library_load():
        if e["id"] == lid:
            return e
    raise HTTPException(404, "not in library")


@app.get("/api/library")
def library_list():
    return {"items": _library_load()}


@app.get("/api/library/{lid}/image")
def library_image(lid: str):
    entry = _library_entry(lid)
    path = os.path.join(_LIBRARY_DIR, f"{lid}.png")
    if not os.path.exists(path):
        raise HTTPException(404, "file missing")
    with open(path, "rb") as f:
        png = f.read()
    return Response(png, media_type="image/png", headers={
        "Content-Disposition": f'inline; filename="{entry["name"]}"'})


@app.get("/api/library/{lid}/thumb")
def library_thumb(lid: str, s: int = 480):
    _library_entry(lid)
    path = os.path.join(_LIBRARY_DIR, f"{lid}.png")
    if not os.path.exists(path):
        raise HTTPException(404, "file missing")
    img = Image.open(path)
    img.thumbnail((s, s))
    return Response(pipeline.to_jpeg_bytes(img, quality=82),
                    media_type="image/jpeg")


@app.delete("/api/library/{lid}")
def library_delete(lid: str):
    with _library_lock:
        entries = [e for e in _library_load() if e["id"] != lid]
        _library_write(entries)
    try:
        os.remove(os.path.join(_LIBRARY_DIR, f"{lid}.png"))
    except OSError:
        pass
    return {"ok": True}


# ── Social ─────────────────────────────────────────────────────────────────────

def _prep_social_upload(raw: bytes) -> bytes:
    """Bound an upload to 2048px and re-encode as JPEG-90.

    The stored bytes are re-decoded on EVERY template render, and the Social
    filmstrip fires 17 renders at once — a raw 12MP phone photo decoding to
    ~36MB RGB per render is what OOM-restarts a 512MB instance. 2048px
    comfortably covers the 1080px top render size."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    if max(img.size) > 2048:
        img.thumbnail((2048, 2048), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@app.post("/api/social/upload")
async def social_upload(file: UploadFile = File(...)):
    _evict_stale()
    raw = await file.read()
    try:
        raw = await run_in_threadpool(_prep_social_upload, raw)
    except Exception:
        raise HTTPException(400, "could not open image")
    img_id = uuid.uuid4().hex[:12]
    _social_imgs[img_id] = {"bytes": raw, "touched": time.time()}
    return {"id": img_id}








# ── Social render (style preview over uploaded photo / placeholder) ────────────

_PLACEHOLDER_ICON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static", "social", "placeholder_icon.png")
_testing_img_cache: dict = {}


def _testing_img() -> bytes:
    """Neutral placeholder base: gray canvas + image glyph (built once).

    Shown until the user uploads their own photo, so nobody mistakes a sample
    photo for their upload. gray-100 backdrop, gray-400 Phosphor image icon.
    """
    if "bytes" not in _testing_img_cache:
        canvas = Image.new("RGB", (1080, 1080), (0xED, 0xED, 0xED))
        try:
            icon = Image.open(_PLACEHOLDER_ICON_PATH).convert("RGBA")
            icon.thumbnail((300, 300), Image.LANCZOS)
            canvas.paste(icon, ((canvas.width - icon.width) // 2,
                                (canvas.height - icon.height) // 2), icon)
        except Exception:
            log.warning("placeholder icon missing — using plain gray base")
        _testing_img_cache["bytes"] = pipeline.to_png_bytes(canvas)
    return _testing_img_cache["bytes"]


def _social_render(style: str, theme: str, ratio: str, title: str,
                   subtitle: str, img: str | None, w: int) -> tuple:
    """Validate params, resolve the base image and render. Shared by the
    preview endpoint and the library-saving download endpoint."""
    if style not in social_engine.TESTING_STYLES:
        raise HTTPException(400, f"unknown style {style}")
    if theme not in social_engine.TESTING_THEMES:
        raise HTTPException(400, f"unknown theme {theme}")
    if ratio not in social_engine.STYLE1_H:
        raise HTTPException(400, f"unknown ratio {ratio}")
    if img:
        rec = _social_imgs.get(img)
        if rec is None:
            raise HTTPException(404, "image not found")
        rec["touched"] = time.time()
        img_bytes = rec["bytes"]
    else:
        img_bytes = _testing_img()
    w = max(96, min(w, 2048))
    with _RENDER_SEM:
        return social_engine.render_testing(style, theme, ratio, img_bytes,
                                            title, subtitle, w)


@app.get("/api/testing/render")
def testing_render(style: str = "style1", theme: str = "green",
                   ratio: str = "1:1", title: str = "", subtitle: str = "",
                   w: int = 720, img: str | None = None):
    """Render one style + theme + ratio with text over the base photo.

    `img` (an id from POST /api/social/upload) swaps in an uploaded photo as
    the base image; without it a neutral placeholder is used. Empty title
    falls back to 'Construction Junction'; empty subtitle is omitted (both
    handled inside the renderer).
    """
    png, truncated = _social_render(style, theme, ratio, title, subtitle, img, w)
    # The URL fully determines the output (uploaded images are immutable per
    # id), so responses are safely browser-cacheable — this keeps thumbnail
    # refetches off the backend while the user types.
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600",
                             "X-Title-Truncated": "1" if truncated else "0"})


@app.get("/api/social/download")
def social_download(style: str = "cover1", theme: str = "green",
                    ratio: str = "1:1", title: str = "", subtitle: str = "",
                    w: int = 1080, img: str | None = None):
    """Full-size social render for download — also lands in the Library,
    like every listing export."""
    png, _ = _social_render(style, theme, ratio, title, subtitle, img, w)
    fname = f"cj_social_{style}_{theme}_{ratio.replace(':', 'x')}.png"
    _library_add(fname, "social", ratio, png)
    return Response(png, media_type="image/png", headers={
        "Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/api/health")
def health():
    return {"ok": True, "photos": len(_photos), "social": len(_social_imgs)}
