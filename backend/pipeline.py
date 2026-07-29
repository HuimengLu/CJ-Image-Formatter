"""CJ Studio listing pipeline — framework-free.

Every function that used to read/write st.session_state now takes and returns
explicit values, so the same code runs under FastAPI (or any host):

    make_listing(img)                    -> (result, warn, subject_mask, subject_rgba, orig_layer)
    compose(result, subject_rgba, ...)   -> final PIL image at the chosen ratio

Model handles are cached per-process with functools.lru_cache (the FastAPI app
runs one process; workers each warm their own copy, same as st.cache_resource).
"""
import colorsys
import functools
import io
import logging
import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

log = logging.getLogger("cj_pipeline")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Inputs whose longest side is below this get an AI upscale (Real-ESRGAN x4)
# BEFORE background removal, so the matting model sees a higher-resolution
# subject and the enlarged result stays sharp when blown up to fill the canvas.
# Gated on the whole image (the subject bbox isn't known until after matting).
_UPSCALE_BELOW = int(os.environ.get("CJ_UPSCALE_BELOW", "900"))
_ESRGAN_ENABLED = os.environ.get("CJ_UPSCALE", "1") != "0"

_RATIO_AR = {"1:1": 1.0, "4:3": 4 / 3, "4:5": 4 / 5}


# ── models (cached per process) ────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _withoutbg_model():
    try:
        from withoutbg import WithoutBG
    except ImportError as exc:
        # This deployment ships without the legacy background-removal deps
        # (withoutbg/onnxruntime) to stay light on RAM. The legacy /api/photos
        # path is inert here; New Listing / Social use the OpenAI pipeline.
        raise RuntimeError(
            "Legacy background removal is not available in this deployment. "
            "Use New Listing (OpenAI pipeline) instead."
        ) from exc

    log.info("Loading withoutbg open-source model...")
    t0 = time.perf_counter()
    try:
        model = WithoutBG.opensource()
    except Exception as exc:
        log.error("Failed to initialise withoutbg model: %s", exc, exc_info=True)
        raise RuntimeError(
            "Could not load the background-removal model. "
            "Check your internet connection — the first run downloads the "
            f"model weights from Hugging Face. Detail: {exc}"
        ) from exc
    log.info("withoutbg model ready in %.1f s", time.perf_counter() - t0)
    return model


@functools.lru_cache(maxsize=1)
def _esrgan_model():
    """Load Real-ESRGAN x4 (via spandrel). Returns (net, device, scale) or None."""
    if not _ESRGAN_ENABLED:
        return None
    try:
        import torch
        from spandrel import ModelLoader
    except Exception as exc:
        log.info("Real-ESRGAN unavailable (%s) — skipping AI upscale.", exc)
        return None

    weight = os.path.join(_BASE_DIR, "models", "RealESRGAN_x4plus.pth")
    if not os.path.exists(weight):
        log.info("Real-ESRGAN weight not found at %s — skipping AI upscale.", weight)
        return None

    try:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model = ModelLoader().load_from_file(weight)
        net = model.model.eval().to(device)
        log.info("Real-ESRGAN x%s ready on %s", model.scale, device)
        return net, device, int(model.scale)
    except Exception as exc:
        log.warning("Failed to load Real-ESRGAN: %s", exc, exc_info=True)
        return None


def _esrgan_upscale(rgb: np.ndarray):
    """Upscale HxWx3 uint8 with Real-ESRGAN; falls back to (rgb, 1)."""
    loaded = _esrgan_model()
    if loaded is None:
        return rgb, 1
    net, device, factor = loaded
    try:
        import torch

        t0 = time.perf_counter()
        x = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
        )
        with torch.no_grad():
            y = net(x).clamp(0, 1)
        if device == "mps":
            torch.mps.synchronize()
        out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        log.info(
            "Real-ESRGAN upscaled %dx%d -> %dx%d in %.2f s",
            rgb.shape[1], rgb.shape[0], out.shape[1], out.shape[0],
            time.perf_counter() - t0,
        )
        return out, factor
    except Exception as exc:
        log.warning("Real-ESRGAN inference failed (%s) — using LANCZOS.", exc)
        return rgb, 1


@functools.lru_cache(maxsize=1)
def load_product_bg():
    """Product-shot backdrop texture (static/product_bg.png|jpg), or None.

    Replaces the flat #FDFDF2 fill behind the cut-out product. Missing file →
    None, and the pipeline falls back to the solid colour.
    """
    for ext in ("png", "jpg", "jpeg"):
        path = os.path.join(_BASE_DIR, "static", f"product_bg.{ext}")
        if os.path.exists(path):
            try:
                return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except Exception:
                return None
    return None


# ── subject extraction ─────────────────────────────────────────────────────────

def _extract_subject(pil_img: Image.Image):
    """Remove background; returns (rgb, alpha) uint8 arrays (soft 0-255 matte)."""
    log.info("Removing background from %dx%d image...", pil_img.width, pil_img.height)
    t0 = time.perf_counter()
    try:
        rgba = _withoutbg_model().remove_background(pil_img.convert("RGB"))
    except RuntimeError:
        raise
    except MemoryError as exc:
        log.error("OOM during withoutbg inference: %s", exc, exc_info=True)
        raise RuntimeError(
            "Not enough memory to process this image. "
            "Try a smaller image (< 4000 x 4000 px)."
        ) from exc
    except Exception as exc:
        log.error("withoutbg remove_background() failed: %s", exc, exc_info=True)
        raise RuntimeError(
            f"Background removal failed ({type(exc).__name__}). "
            "Try a different image or restart the app."
        ) from exc
    log.info("Background removed in %.2f s", time.perf_counter() - t0)
    arr = np.array(rgba.convert("RGBA"))
    return arr[:, :, :3], arr[:, :, 3]


def _alpha_binary(alpha: np.ndarray) -> np.ndarray:
    """Threshold + open/close — the speck-cleaned solid silhouette (no hole fill)."""
    _, binary = cv2.threshold(alpha, 15, 255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=2)
    return binary


def _interior_holes(binary: np.ndarray):
    """Connected-component labels of the background + the non-border hole ids.

    Interior holes are enclosed gaps inside the silhouette — a matting miss on a
    solid object, or a *real* cut-out (mesh, handle) on a see-through one.
    """
    inv = cv2.bitwise_not(binary)
    _, labels = cv2.connectedComponents(inv, connectivity=8)
    border_labels = set(np.concatenate([
        labels[0, :], labels[-1, :],
        labels[1:-1, 0], labels[1:-1, -1],
    ]).tolist())
    interior = [int(l) for l in np.unique(labels)
                if l != 0 and int(l) not in border_labels]
    return labels, interior


def _finalize_alpha(filled: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Restore the soft matte inside the kept silhouette, hard-zero the rest."""
    out = np.where(
        filled > 0,
        np.where(alpha > 0, alpha, np.uint8(255)),
        np.uint8(0),
    ).astype(np.uint8)
    out[out < 80] = 0
    return out


def _tighten_alpha(alpha: np.ndarray) -> np.ndarray:
    """Speck cleanup, enclosed-hole fill, hard-zero low-confidence pixels."""
    binary = _alpha_binary(alpha)
    labels, interior = _interior_holes(binary)
    filled = binary.copy()
    for lbl in interior:
        filled[labels == lbl] = 255
    return _finalize_alpha(filled, alpha)


def _hole_area_cap(binary: np.ndarray) -> int:
    """Max hole area (px) still considered a matting miss: 0.3% of the subject."""
    return max(64, int(np.count_nonzero(binary) * 0.003))


def _tighten_alpha_small_holes(alpha: np.ndarray) -> np.ndarray:
    """Like _tighten_alpha, but only fills holes below the area cap.

    Larger enclosed gaps are treated as real cut-outs (mesh, handles) and stay
    transparent so the new backdrop shows through. Production strategy for the
    Listing workflow; Testing 2 uses the same matte for every shadow panel.
    """
    binary = _alpha_binary(alpha)
    labels, interior = _interior_holes(binary)
    area_cap = _hole_area_cap(binary)
    filled = binary.copy()
    for lbl in interior:
        if int(np.count_nonzero(labels == lbl)) < area_cap:
            filled[labels == lbl] = 255
    return _finalize_alpha(filled, alpha)


def _estimate_bg_color(rgb: np.ndarray):
    """Median colour of the four border strips — the background colour, or None."""
    h, w = rgb.shape[:2]
    b = max(10, min(30, h // 20, w // 20))
    strips = [rgb[:b, :], rgb[-b:, :], rgb[:, :b], rgb[:, -b:]]
    border_px = np.concatenate([s.reshape(-1, 3) for s in strips]).astype(np.float32)
    if len(border_px) == 0:
        return None
    return np.median(border_px, axis=0)


def _color_guided_cleanup(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Zero moderate-alpha pixels whose colour matches the background colour."""
    bg_color = _estimate_bg_color(rgb)
    if bg_color is None or float(np.mean(bg_color)) > 230:
        return alpha
    diff = rgb.astype(np.float32) - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))
    out = alpha.copy()
    uncertain = (alpha >= 80) & (alpha <= 210)
    out[uncertain & (dist < 45)] = 0
    return out


# ── product colour grade ────────────────────────────────────────────────────────
#
# Sliders lifted from the supplied Lightroom preset (Downloads/preset.xmp,
# "FindPreset Generated"). We approximate Camera Raw's render — exact parity
# needs Adobe's engine — and apply the tonal + colour look to the cut-out
# subject *after* background removal, *before* it lands on the new backdrop.
_PRESET = dict(
    temp=5800, tint=3,
    exposure=0.25, contrast=5,
    highlights=-22, shadows=18, whites=-8, blacks=-5,
    texture=14, clarity=5,
    vibrance=-8, saturation=-4,
    p_shadow=5, p_dark=6, p_light=4, p_high=-8,   # parametric tone curve
    grade=[                                       # (hue°, sat, tonal region)
        (30, 6, "shadow"),
        (32, 4, "mid"),
        (35, 3, "high"),
        (30, 2, "global"),
    ],
    grade_blend=50,
)

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def _hue_dir(hue: float) -> np.ndarray:
    """Chroma push (R,G,B around grey) for a colour-grade hue in degrees."""
    r, g, b = colorsys.hsv_to_rgb((hue % 360) / 360.0, 1.0, 1.0)
    v = np.array([r, g, b], np.float32)
    return v - v.mean()


def _apply_color_grade(rgb: np.ndarray) -> np.ndarray:
    """Approximate the Lightroom preset on an HxWx3 uint8 RGB subject."""
    p = _PRESET
    img = rgb.astype(np.float32) / 255.0

    # 1. White balance — subtle warm / green-magenta nudge (Kelvin approximated).
    warm = (p["temp"] - 5500) / 5500.0
    m = p["tint"] / 100.0
    img[..., 0] *= 1.0 + 0.35 * warm + 0.05 * m
    img[..., 1] *= 1.0 - 0.03 * m
    img[..., 2] *= 1.0 - 0.35 * warm + 0.05 * m
    img = np.clip(img, 0.0, 1.0)

    # 2. Exposure (stops).
    img = np.clip(img * (2.0 ** p["exposure"]), 0.0, 1.0)

    # 3. Contrast — gentle S-curve around mid grey.
    c = p["contrast"] / 100.0
    img = np.clip((img - 0.5) * (1.0 + c) + 0.5, 0.0, 1.0)

    # 4. Highlights / Shadows / Whites / Blacks via luminance-masked offsets.
    lum = img @ _LUMA
    adj = (
        (p["shadows"] / 100.0) * 0.5 * (1.0 - lum) ** 2
        + (p["highlights"] / 100.0) * 0.5 * lum ** 2
        + (p["blacks"] / 100.0) * 0.3 * (1.0 - lum) ** 3
        + (p["whites"] / 100.0) * 0.3 * lum ** 3
    )
    img = np.clip(img + adj[..., None], 0.0, 1.0)

    # 5. Parametric tone curve — four triangular tonal bands.
    lum = img @ _LUMA
    padj = sum(
        (p[key] / 100.0) * 0.25 * np.clip(1.0 - np.abs(lum - c) / 0.125, 0, 1)
        for key, c in (("p_shadow", 0.125), ("p_dark", 0.375),
                       ("p_light", 0.625), ("p_high", 0.875))
    )
    img = np.clip(img + padj[..., None], 0.0, 1.0)

    # 6. Texture / Clarity — mild local-contrast pop on the mid-frequencies.
    amount = (p["texture"] + p["clarity"]) / 100.0 * 0.5
    if amount > 0:
        blur = cv2.GaussianBlur(img, (0, 0), 3.0)
        img = np.clip(img + (img - blur) * amount, 0.0, 1.0)

    # 7. Vibrance / Saturation in HSV (vibrance protects already-saturated px).
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1]
    s *= 1.0 + p["saturation"] / 100.0
    s *= 1.0 + (p["vibrance"] / 100.0) * (1.0 - s)
    hsv[..., 1] = np.clip(s, 0.0, 1.0)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 8. Colour grading — warm tint weighted per tonal region.
    lum = img @ _LUMA
    region = {
        "shadow": (1.0 - lum) ** 2,
        "mid": 1.0 - (2.0 * lum - 1.0) ** 2,
        "high": lum ** 2,
        "global": np.ones_like(lum),
    }
    blend = p["grade_blend"] / 100.0
    tint = np.zeros_like(img)
    for hue, sat, reg in p["grade"]:
        tint += _hue_dir(hue)[None, None, :] * (sat / 100.0) * 0.5 * region[reg][..., None]
    img = np.clip(img + tint * blend, 0.0, 1.0)

    return (img * 255.0 + 0.5).astype(np.uint8)


def generate_product_shadow(r_mask, bx_s, by_s, bw_s, bh_s, ox, oy, target_cx, CS):
    """Simple soft drop shadow: blurred silhouette shifted slightly left/down.

    `CS` may be an int (square canvas) or a (height, width) tuple.
    """
    H, W = r_mask.shape[:2]
    CH, CW = (CS, CS) if isinstance(CS, int) else CS
    empty = np.zeros((CH, CW), dtype=np.float32)

    if int(np.count_nonzero(r_mask > 128)) < 100 or bw_s < 4 or bh_s < 4:
        return empty, empty

    sh_x = int(bw_s * -0.03)          # negative → shadow shifts left
    sh_y = max(1, int(bh_s * 0.02))

    shadow = empty.copy()
    s_oy, s_ox = oy + sh_y, ox + sh_x
    cy0 = max(0, s_oy); cy1 = min(CH, s_oy + H)
    cx0 = max(0, s_ox); cx1 = min(CW, s_ox + W)
    if cy1 > cy0 and cx1 > cx0:
        iy0 = cy0 - s_oy; iy1 = iy0 + (cy1 - cy0)
        ix0 = cx0 - s_ox; ix1 = ix0 + (cx1 - cx0)
        shadow[cy0:cy1, cx0:cx1] = r_mask[iy0:iy1, ix0:ix1].astype(np.float32) / 255.0

    shadow = cv2.GaussianBlur(shadow, (0, 0), 24.0)
    shadow = np.clip(shadow * 0.16, 0.0, 1.0)
    return shadow, empty


def _place_subject(rgb: np.ndarray, alpha: np.ndarray):
    """Crop to the subject, colour-grade it, and fit onto the 1600² backdrop.

    Returns (out_pil, full_mask, full_rgba, full_orig), or None when no
    usable subject remains.
    """
    h, w = rgb.shape[:2]
    total = h * w

    fg_px = int(np.count_nonzero(alpha > 15))
    if fg_px < total * 0.01:
        return None

    pts = cv2.findNonZero((alpha > 15).astype(np.uint8) * 255)
    if pts is None:
        return None
    bx, by, bw, bh = cv2.boundingRect(pts)

    pad = max(10, int(max(bw, bh) * 0.04))
    x1, y1 = max(0, bx - pad), max(0, by - pad)
    x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
    crop_rgb = rgb[y1:y2, x1:x2]
    crop_alpha = alpha[y1:y2, x1:x2]

    # Colour grading disabled — keep the subject's native colours.

    CS = 1600
    bg_tex = load_product_bg()
    if bg_tex is not None:
        canvas = np.array(ImageOps.fit(bg_tex, (CS, CS), Image.LANCZOS))
    else:
        canvas = np.full((CS, CS, 3), [253, 253, 242], dtype=np.uint8)  # #FDFDF2

    # Subject was already AI-upscaled before matting (if small); here we only
    # fit it to the canvas. bw/bh are in the (possibly upscaled) resolution, so
    # `scale` shrinks to match and the final on-canvas size is unchanged.
    scale = (CS * 0.75) / max(bw, bh)
    ch_c, cw_c = crop_rgb.shape[:2]
    nw = max(1, int(cw_c * scale))
    nh = max(1, int(ch_c * scale))
    r_rgb = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    r_mask = cv2.resize(crop_alpha, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    scx = int((bx - x1 + bw / 2) * scale)
    scy = int((by - y1 + bh / 2) * scale)

    target_cx = CS // 2
    target_cy = int(CS * 0.52)
    ox = target_cx - scx
    oy = target_cy - scy

    def _slices(offset: int, length: int, limit: int):
        c0 = max(0, offset)
        c1 = min(limit, offset + length)
        return slice(c0, c1), slice(c0 - offset, c1 - offset)

    can_y, img_y = _slices(oy, nh, CS)
    can_x, img_x = _slices(ox, nw, CS)

    bx_s = int((bx - x1) * scale)
    by_s = int((by - y1) * scale)
    bw_s = int(bw * scale)
    bh_s = int(bh * scale)

    cast_shadow, contact_shadow = generate_product_shadow(
        r_mask, bx_s, by_s, bw_s, bh_s, ox, oy, target_cx, CS,
    )

    total_shadow = np.clip(cast_shadow + contact_shadow, 0.0, 1.0)
    canvas_f = canvas.astype(np.float32)
    for c in range(3):
        canvas_f[:, :, c] = np.clip(canvas_f[:, :, c] * (1.0 - total_shadow), 0, 255)
    canvas = canvas_f.astype(np.uint8)

    alpha_f = r_mask[img_y, img_x].astype(np.float32) / 255.0
    roi = canvas[can_y, can_x].astype(np.float32)
    for c in range(3):
        roi[:, :, c] = r_rgb[img_y, img_x, c] * alpha_f + roi[:, :, c] * (1.0 - alpha_f)
    canvas[can_y, can_x] = roi.astype(np.uint8)

    full_mask = np.zeros((CS, CS), dtype=np.uint8)
    full_mask[can_y, can_x] = r_mask[img_y, img_x]

    full_rgba = np.zeros((CS, CS, 4), dtype=np.uint8)
    full_rgba[can_y, can_x, :3] = r_rgb[img_y, img_x]
    full_rgba[can_y, can_x, 3] = r_mask[img_y, img_x]

    # Whole (post-upscale) photo mapped onto the canvas with the subject's exact
    # transform — crop origin (x1, y1), uniform `scale`, paste offset (ox, oy) —
    # expressed as one affine so the original registers pixel-perfectly under the
    # cut-out (drives the "Original Background" 25% preview layer). The alpha
    # channel marks where the photo actually covers the canvas.
    M = np.float32([[scale, 0, ox - x1 * scale], [0, scale, oy - y1 * scale]])
    full_orig = np.dstack([
        cv2.warpAffine(rgb, M, (CS, CS), flags=cv2.INTER_LINEAR),
        cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (CS, CS),
                       flags=cv2.INTER_LINEAR),
    ])

    out = Image.fromarray(canvas)
    out = ImageEnhance.Brightness(out).enhance(1.02)
    out = ImageEnhance.Contrast(out).enhance(1.05)
    return out, full_mask, full_rgba, full_orig


def make_listing(pil_img: Image.Image):
    """Full pipeline: subject extraction → crop → 1600x1600 canvas + shadow.

    Returns (result_pil, warn, subject_mask, subject_rgba, orig_layer); on
    failure the first element is None and warn holds the message. orig_layer is
    the whole photo mapped onto the canvas with the subject's exact transform
    (see _place_subject) for the "Original Background" preview toggle.
    """
    log.info("make_listing: start input=%dx%d", pil_img.width, pil_img.height)
    t_total = time.perf_counter()

    # AI-upscale small inputs up front, so background removal (and everything
    # downstream) runs on a higher-resolution subject. Gated on the whole
    # image's longest side, since the subject size isn't known until matting.
    if max(pil_img.width, pil_img.height) < _UPSCALE_BELOW:
        up_rgb, up = _esrgan_upscale(np.array(pil_img.convert("RGB")))
        if up > 1:
            pil_img = Image.fromarray(up_rgb)
            log.info("make_listing: pre-upscaled input -> %dx%d",
                     pil_img.width, pil_img.height)

    try:
        rgb, alpha = _extract_subject(pil_img)
    except RuntimeError as exc:
        log.warning("make_listing: subject extraction failed - %s", exc)
        return None, str(exc), None, None, None

    # B2 hole strategy: small enclosed gaps are matting misses (filled); larger
    # ones are real cut-outs — mesh, handles — kept open for the new backdrop.
    alpha = _tighten_alpha_small_holes(alpha)
    alpha = _color_guided_cleanup(rgb, alpha)

    placed = _place_subject(rgb, alpha)
    if placed is None:
        return None, "Could not detect the item. Try a photo with a cleaner background.", None, None, None
    out, full_mask, full_rgba, full_orig = placed
    log.info(
        "make_listing: done in %.2f s output=%dx%d",
        time.perf_counter() - t_total, out.width, out.height,
    )
    return out, None, full_mask, full_rgba, full_orig


# ── text overlay + composition ─────────────────────────────────────────────────

def _load_font(size: int, style: str) -> ImageFont.ImageFont:
    """Try to load a suitable system font for each style, fall back to default."""
    candidates = {
        "Bold & Loud": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/Library/Fonts/Helvetica Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        ],
        "Clean & Minimal": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Helvetica.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ],
    }
    for path in candidates.get(style, []):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _split_words(words: list, max_lines: int) -> list:
    if not words:
        return [""]
    n = min(len(words), max_lines)
    size = math.ceil(len(words) / n)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def _render_background_text(canvas: Image.Image, CS: int) -> Image.Image:
    """Two-line Construction Junction text layer (fallback when no artwork)."""
    lines = ["CONSTRUCTION", "JUNCTION"]
    layer = Image.new("RGBA", (CS, CS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    font_size = int(CS * 0.18)
    font = _load_font(font_size, "Bold & Loud")
    for _ in range(20):
        widths = [draw.textlength(line, font=font) for line in lines]
        if max(widths) < CS * 0.96:
            break
        font_size = max(140, int(font_size * 0.94))
        font = _load_font(font_size, "Bold & Loud")

    bbox = font.getbbox("Ay")
    line_height = bbox[3] - bbox[1] + int(font_size * 0.08)
    y = int(CS * 0.08)
    for line in lines:
        width = draw.textlength(line, font=font)
        x = (CS - width) // 2
        draw.text((x, y), line, font=font, fill=(210, 235, 95, 180))
        y += line_height

    out = canvas.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def apply_text_overlay(base: Image.Image, subject_rgba: np.ndarray,
                       bg_image: Image.Image, target_ratio: str = "1:1",
                       description_text: str = "") -> tuple:
    """Artwork (background) + shadow + scaled-down centred subject.

    Returns (result, subject_bottom_y) — the bottom edge feeds the caption
    positioning in add_description_text (state that used to ride session).
    """
    CS = base.width  # 1600
    bg_color = (253, 253, 242)
    canvas_img = Image.new("RGBA", (CS, CS), tuple(bg_color) + (255,))

    artwork_w, artwork_h = 0, 0
    if bg_image is not None:
        bw, bh = bg_image.size
        ar = _RATIO_AR.get(target_ratio, 1.0)
        if ar >= 1.0:
            safe_width, safe_height = CS, int(CS / ar)
        else:
            safe_width, safe_height = int(CS * ar), CS
        safe_margin = 0.02
        max_w = int(safe_width * (1.0 - 2 * safe_margin))
        max_h = int(safe_height * (1.0 - 2 * safe_margin))
        scale = min(max_w / float(bw), max_h / float(bh))
        artwork_w = max(1, int(bw * scale))
        artwork_h = max(1, int(bh * scale))

    subject_w, subject_h = 0, 0
    if subject_rgba is not None:
        scale_factor = 0.94
        subject_h, subject_w = subject_rgba.shape[:2]
        subject_w = max(1, int(subject_w * scale_factor))
        subject_h = max(1, int(subject_h * scale_factor))

    desc_height = 0
    if description_text.strip():
        font_size = max(32, int(CS * 0.04 * 0.4))
        line_height = int(font_size * 1.2)
        max_width = int(CS * 0.9)
        word_count = len(description_text.split())
        avg_chars_per_line = max(3, max_width // (font_size * 0.6))
        estimated_words_per_line = max(1, int(avg_chars_per_line / 5))
        estimated_lines = max(1, (word_count + estimated_words_per_line - 1)
                              // estimated_words_per_line)
        desc_height = int(line_height * estimated_lines + CS * 0.03)

    artwork_gap = int(CS * 0.03)
    subject_gap = int(CS * 0.03)
    total_comp_height = artwork_h + artwork_gap + subject_h + subject_gap + desc_height
    available_space = CS - total_comp_height
    vertical_offset = max(0, available_space // 2)

    artwork_bottom_y = 0
    if bg_image is not None:
        bw, bh = bg_image.size
        ar = _RATIO_AR.get(target_ratio, 1.0)
        if ar >= 1.0:
            safe_width, safe_height = CS, int(CS / ar)
        else:
            safe_width, safe_height = int(CS * ar), CS
        safe_margin = 0.02
        max_w = int(safe_width * (1.0 - 2 * safe_margin))
        max_h = int(safe_height * (1.0 - 2 * safe_margin))
        scale = min(max_w / float(bw), max_h / float(bh))
        new_w = max(1, int(bw * scale))
        new_h = max(1, int(bh * scale))
        try:
            resized = bg_image.resize((new_w, new_h), Image.LANCZOS)
        except Exception:
            resized = bg_image.resize((new_w, new_h))
        rgba = resized.convert("RGBA")
        x = (CS - new_w) // 2
        y = vertical_offset + max(int(CS * 0.02), int(safe_height * safe_margin))
        canvas_img.paste(rgba, (x, y), rgba)
        artwork_bottom_y = y + new_h
    else:
        base_rgb = np.full((CS, CS, 3), bg_color, dtype=np.uint8)
        text_art = _render_background_text(Image.fromarray(base_rgb), CS)
        text_rgba = text_art.convert("RGBA")
        canvas_img.paste(text_rgba, (0, 0), text_rgba)

    canvas = np.array(canvas_img.convert("RGB"))

    if subject_rgba is None:
        result = Image.fromarray(canvas)
        result = ImageEnhance.Brightness(result).enhance(1.02)
        result = ImageEnhance.Contrast(result).enhance(1.05)
        return result, CS

    scale_factor = 0.94
    subject_h, subject_w = subject_rgba.shape[:2]
    new_w = max(1, int(subject_w * scale_factor))
    new_h = max(1, int(subject_h * scale_factor))

    scaled_rgb = cv2.resize(subject_rgba[:, :, :3], (new_w, new_h),
                            interpolation=cv2.INTER_LANCZOS4)
    scaled_alpha = cv2.resize(subject_rgba[:, :, 3], (new_w, new_h),
                              interpolation=cv2.INTER_LANCZOS4)

    center_x = CS // 2
    ox = center_x - new_w // 2

    if artwork_bottom_y > 0:
        oy = artwork_bottom_y + int(CS * 0.03)
    else:
        center_y = CS // 2
        oy = center_y - new_h // 2
        oy += vertical_offset - int(CS * 0.04)
    oy = max(0, min(oy, CS - new_h))

    cast_shadow, _ = generate_product_shadow(
        scaled_alpha, 0, 0, new_w, new_h, ox, oy, center_x, CS)

    total_shadow = np.clip(cast_shadow, 0.0, 1.0)
    canvas_f = canvas.astype(np.float32)
    for c in range(3):
        canvas_f[:, :, c] = np.clip(canvas_f[:, :, c] * (1.0 - total_shadow), 0, 255)
    canvas = canvas_f.astype(np.uint8)

    def _slices(offset: int, length: int, limit: int):
        c0 = max(0, offset)
        c1 = min(limit, offset + length)
        return slice(c0, c1), slice(c0 - offset, c1 - offset)

    can_y, img_y = _slices(oy, new_h, CS)
    can_x, img_x = _slices(ox, new_w, CS)

    alpha_f = scaled_alpha[img_y, img_x].astype(np.float32) / 255.0
    roi = canvas[can_y, can_x].astype(np.float32)
    for c in range(3):
        roi[:, :, c] = scaled_rgb[img_y, img_x, c] * alpha_f + roi[:, :, c] * (1.0 - alpha_f)
    canvas[can_y, can_x] = roi.astype(np.uint8)

    subject_bottom = can_y.stop if hasattr(can_y, "stop") else (oy + new_h)

    result = Image.fromarray(canvas)
    result = ImageEnhance.Brightness(result).enhance(1.02)
    result = ImageEnhance.Contrast(result).enhance(1.05)
    return result, subject_bottom


def add_description_text(img: Image.Image, text: str, subject_bottom_y: int) -> Image.Image:
    """Small grey caption under the product (bottom edge from apply_text_overlay)."""
    if not text.strip():
        return img

    W, H = img.size
    draw = ImageDraw.Draw(img)
    font_size = max(32, int(W * 0.04))
    font = _load_font(font_size, "Clean & Minimal")

    max_width = int(W * 0.9)
    lines = []
    current_line = ""
    for word in text.split():
        test_line = current_line + (" " if current_line else "") + word
        if draw.textlength(test_line, font=font) > max_width:
            if current_line:
                lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    line_height = int(font_size * 1.2)
    total_height = line_height * len(lines)

    full_size = 1600
    scale_factor = H / float(full_size) if H < full_size else 1.0
    adjusted_subject_bottom = int(subject_bottom_y * scale_factor)

    gap = int(W * 0.03)
    y = min(adjusted_subject_bottom + gap, H - total_height - int(W * 0.02))
    y = max(y, int(W * 0.01))

    for line in lines:
        width = draw.textlength(line, font=font)
        x = (W - width) // 2
        draw.text((x, y), line, font=font, fill=(100, 100, 100, 255))
        y += line_height

    return img


def _corner_color(img: Image.Image):
    """Average the four corner pixels — the padding colour, so bars are seamless."""
    w, h = img.size
    px = img.load()
    pts = [(1, 1), (w - 2, 1), (1, h - 2), (w - 2, h - 2)]
    cols = [px[x, y] for x, y in pts]
    n = len(cols[0]) if isinstance(cols[0], tuple) else 1
    if n == 1:
        return int(round(sum(cols) / len(cols)))
    return tuple(int(round(sum(c[i] for c in cols) / len(cols))) for i in range(n))


def fit_to_ratio(img: Image.Image, ratio: str) -> Image.Image:
    """Pad img with its own background colour to the target aspect ratio."""
    W, H = img.size
    ar = _RATIO_AR.get(ratio, 1.0)
    cur = W / float(H)
    if abs(cur - ar) < 1e-3:
        return img
    if ar > cur:
        new_w, new_h = int(round(H * ar)), H
    else:
        new_w, new_h = W, int(round(W / ar))
    canvas = Image.new(img.mode, (new_w, new_h), _corner_color(img))
    canvas.paste(img, ((new_w - W) // 2, (new_h - H) // 2))
    return canvas


def cover_to_ratio(img: Image.Image, ratio: str) -> Image.Image:
    """Cover-fit img to the target ratio: scale to fill, centre-crop overflow.

    Used for Cover-mode scenes — the generated environment should fill the
    frame edge-to-edge (no padded bars), unlike fit_to_ratio's padding.
    """
    return ImageOps.fit(img, _ratio_dims(ratio), Image.LANCZOS)


def _ratio_dims(ratio: str, base: int = 1600) -> tuple:
    """(W, H) for the target ratio, with the shorter side fixed at `base`."""
    ar = _RATIO_AR.get(ratio, 1.0)
    if ar >= 1.0:                       # wider than tall (e.g. 4:3)
        return int(round(base * ar)), base
    return base, int(round(base / ar))  # taller than wide (e.g. 4:5)


def compose_fullbleed(subject_rgba, ratio: str, orig_layer=None) -> Image.Image:
    """Target-ratio image whose backdrop texture covers the whole frame (no
    solid bars), with the cut-out subject + soft shadow centred on top.

    The texture is cover-fit straight to the target dimensions, so 4:3 / 4:5
    fill edge-to-edge instead of padding a 1:1 square.

    `orig_layer` (RGBA, canvas space of subject_rgba) is the whole original
    photo already carrying the subject's exact transform; when given it is
    blended at 25% opacity between the backdrop and the subject, shifted by the
    same translation the subject gets here — never re-scaled — so the two stay
    registered.
    """
    W, H = _ratio_dims(ratio)
    bg = load_product_bg()
    if bg is not None:
        canvas = np.array(ImageOps.fit(bg, (W, H), Image.LANCZOS))
    else:
        canvas = np.full((H, W, 3), [253, 253, 242], dtype=np.uint8)  # #FDFDF2

    if subject_rgba is not None and int(np.count_nonzero(subject_rgba[:, :, 3] > 15)):
        ys, xs = np.where(subject_rgba[:, :, 3] > 15)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = subject_rgba[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        ox = W // 2 - cw // 2
        oy = int(H * 0.52) - ch // 2    # same vertical anchor as make_listing

        def _slices(offset: int, length: int, limit: int):
            c0 = max(0, offset); c1 = min(limit, offset + length)
            return slice(c0, c1), slice(c0 - offset, c1 - offset)

        # Original Background layer: same translation as the subject (ox - x0,
        # oy - y0), blended at 25% before shadow + subject go on top.
        if orig_layer is not None:
            ocy, oiy = _slices(oy - y0, orig_layer.shape[0], H)
            ocx, oix = _slices(ox - x0, orig_layer.shape[1], W)
            if ocy.stop > ocy.start and ocx.stop > ocx.start:
                oa = orig_layer[oiy, oix, 3].astype(np.float32) / 255.0 * 0.25
                roi = canvas[ocy, ocx].astype(np.float32)
                for c in range(3):
                    roi[:, :, c] = orig_layer[oiy, oix, c] * oa + roi[:, :, c] * (1.0 - oa)
                canvas[ocy, ocx] = roi.astype(np.uint8)

        cast_shadow, _ = generate_product_shadow(
            crop[:, :, 3], 0, 0, cw, ch, ox, oy, W // 2, (H, W))
        canvas_f = canvas.astype(np.float32)
        for c in range(3):
            canvas_f[:, :, c] = np.clip(canvas_f[:, :, c] * (1.0 - cast_shadow), 0, 255)
        canvas = canvas_f.astype(np.uint8)

        can_y, img_y = _slices(oy, ch, H)
        can_x, img_x = _slices(ox, cw, W)
        alpha_f = crop[img_y, img_x, 3].astype(np.float32) / 255.0
        roi = canvas[can_y, can_x].astype(np.float32)
        for c in range(3):
            roi[:, :, c] = crop[img_y, img_x, c] * alpha_f + roi[:, :, c] * (1.0 - alpha_f)
        canvas[can_y, can_x] = roi.astype(np.uint8)

    out = Image.fromarray(canvas)
    out = ImageEnhance.Brightness(out).enhance(1.02)
    out = ImageEnhance.Contrast(out).enhance(1.05)
    return out


def compose(result: Image.Image, subject_rgba, ratio: str,
            text_mode: bool, caption: str, orig_layer=None) -> Image.Image:
    """Final image for one photo at its chosen ratio + text settings.

    Text mode keeps the 1:1 artwork composition (padded to ratio); the plain
    product shot is rebuilt at the target ratio so the backdrop fills the frame.
    `orig_layer` (Original Background toggle) applies to the plain path only.
    """
    if text_mode:
        # No artwork layer since bg_artwork.png was dropped — apply_text_overlay
        # falls back to its rendered-text background on None.
        square, subject_bottom = apply_text_overlay(
            result, subject_rgba, None, "1:1", caption)
        if caption.strip():
            square = add_description_text(square, caption.strip(), subject_bottom)
        return fit_to_ratio(square, ratio)
    return compose_fullbleed(subject_rgba, ratio, orig_layer)


# ── Testing 2: OpenAI white-frame + multiply composition ────────────────────────

# Per-ratio zoom applied to the model frame before centring on the backdrop.
# 4:5 enlarges 20% so the product fills that taller ratio; others contain-fit.
_TESTING2_ZOOM = {"4:5": 1.2}

# Cover mode: category → scene background in static/cover/. Explicit mapping —
# the filenames don't follow a strict "<category> Background" rule (Building
# Materials/Outdoor ships as just "Outdoor", since a "/" can't go in a filename).
_COVER_BG = {
    "Furniture": "Furniture Background.png",
    "Appliances & Fixtures": "Appliances & Fixtures Background.png",
    "Building Materials/Outdoor": "Outdoor Background.png",
    "Lighting": "Lighting Background.png",
    "Specialty": "Specialty Background.png",
}


@functools.lru_cache(maxsize=8)
def _load_cover_bg(category: str):
    """Scene background PNG for one cover category, or None if missing."""
    name = _COVER_BG.get(category)
    if not name:
        return None
    path = os.path.join(_BASE_DIR, "static", "cover", name)
    if not os.path.exists(path):
        return None
    try:
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:
        return None


def make_cover(pil_img: Image.Image):
    """Cover mode: classify the product, then compose it into that category's
    scene via gpt-image-2 (original photo + scene background + fixed prompt).

    Bypasses the white-plate/multiply path entirely — the model's scene output
    IS the result. Returns (cover_pil_rgb, category, warn); on failure the
    first element is None.
    """
    from . import openai_engine

    log.info("make_cover: start input=%dx%d", pil_img.width, pil_img.height)
    t_total = time.perf_counter()

    category = openai_engine.classify_category(pil_img)
    bg = _load_cover_bg(category)
    if bg is None:
        return None, category, f"cover background missing for {category}"

    scene, warn = openai_engine.cover_scene(pil_img, bg, category)
    if scene is None:
        return None, category, warn

    log.info("make_cover: done in %.2f s category=%s out=%dx%d",
             time.perf_counter() - t_total, category, scene.width, scene.height)
    return scene.convert("RGB"), category, None


def make_listing_openai(pil_img: Image.Image):
    """Testing-2 builder: OpenAI white-bg + studio shadow → full frame for multiply.

    Flow: gpt-image-2 remove-bg/shadow → optional ESRGAN upscale. The whole
    model output is kept — no crop, no exposure/contrast/white edits — so its
    centred composition and balanced whitespace survive intact; compose_testing2
    contain-fits and centres it on the backdrop. Returns (frame_rgb, warn), an
    HxWx3 uint8 white-background image, or (None, warn) on failure.
    """
    from . import openai_engine

    log.info("make_listing_openai: start input=%dx%d", pil_img.width, pil_img.height)
    t_total = time.perf_counter()

    white, warn = openai_engine.white_bg_with_shadow(pil_img)
    if white is None:
        return None, warn or "OpenAI could not process this image."

    rgb = np.array(white.convert("RGB"))
    if max(rgb.shape[:2]) < _UPSCALE_BELOW and _ESRGAN_ENABLED:
        rgb, _ = _esrgan_upscale(rgb)

    log.info("make_listing_openai: done in %.2f s frame=%dx%d",
             time.perf_counter() - t_total, rgb.shape[1], rgb.shape[0])
    return rgb, None


def compose_testing2(plate: np.ndarray, ratio: str) -> Image.Image:
    """Multiply the full gpt-image-2 frame onto the CJ product backdrop.

    The whole model frame is contain-fitted (all of its content stays visible)
    and centred on the backdrop, then multiply-blended — pure-white areas leave
    the backdrop untouched, only the product and its shadow darken it. Because
    the model already centres the product with balanced whitespace, centring the
    frame keeps the product centred while preserving its full composition.

    4:5 gets a 20% enlargement (`_TESTING2_ZOOM`) so the product isn't dwarfed
    by that ratio's extra height; the enlarged frame is centred with the
    overflow clipped to the canvas.
    """
    W, H = _ratio_dims(ratio)
    bg = load_product_bg()
    if bg is not None:
        canvas = np.array(ImageOps.fit(bg, (W, H), Image.LANCZOS))
    else:
        canvas = np.full((H, W, 3), [253, 253, 242], dtype=np.uint8)  # #FDFDF2

    ph, pw = plate.shape[:2]
    s = min(W / pw, H / ph) * _TESTING2_ZOOM.get(ratio, 1.0)   # contain-fit × per-ratio zoom
    nw, nh = max(1, round(pw * s)), max(1, round(ph * s))
    if (nw, nh) != (pw, ph):
        plate = cv2.resize(plate, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # Centre the (possibly oversized) frame, clipping any overflow to the canvas.
    ox, oy = (W - nw) // 2, (H - nh) // 2
    cy0, cy1 = max(0, oy), min(H, oy + nh)
    cx0, cx1 = max(0, ox), min(W, ox + nw)
    py0, px0 = cy0 - oy, cx0 - ox
    top = plate[py0:py0 + (cy1 - cy0), px0:px0 + (cx1 - cx0)].astype(np.float32)
    base = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    canvas[cy0:cy1, cx0:cx1] = np.clip(base * top / 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(canvas)


def draw_dimensions(img: Image.Image, dims: list[dict]) -> Image.Image:
    """Burn dimension annotations into an export image.

    Mirrors DimensionOverlay.tsx: coordinates are normalized (0-1) over the
    composed image; each dimension is a main line extended slightly past both
    endpoints, perpendicular end caps, and the value label offset perpendicular
    ("up") from the midpoint. The overlay's px constants (stroke 1.5, ext 8,
    cap 5, gap 22, font 18) are relative to a ~760px on-screen preview, so they
    scale with image width here. Drawn at 2x on a transparent layer and
    downsampled for antialiasing.
    """
    if not dims:
        return img
    w, h = img.size
    S = 2
    layer = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    color = (93, 94, 102, 255)                      # #5d5e66, same as the overlay
    scale = (w / 760) * S
    lw = max(2, round(1.5 * scale))
    ext, cap, gap = 8 * scale, 5 * scale, 22 * scale
    font = _load_font(max(12, round(18 * scale)), "Clean & Minimal")

    for d in dims:
        ax, ay = d["start"]["x"] * w * S, d["start"]["y"] * h * S
        bx, by = d["end"]["x"] * w * S, d["end"]["y"] * h * S
        dx, dy = bx - ax, by - ay
        ln = math.hypot(dx, dy) or 1.0
        ux, uy = dx / ln, dy / ln
        a2 = (ax - ux * ext, ay - uy * ext)
        b2 = (bx + ux * ext, by + uy * ext)
        cvx, cvy = -uy * cap, ux * cap
        nx, ny = -uy, ux
        if ny > 0:                                   # perpendicular pointing "up"
            nx, ny = -nx, -ny

        draw.line([a2, b2], fill=color, width=lw)
        draw.line([(a2[0] - cvx, a2[1] - cvy), (a2[0] + cvx, a2[1] + cvy)],
                  fill=color, width=lw)
        draw.line([(b2[0] - cvx, b2[1] - cvy), (b2[0] + cvx, b2[1] + cvy)],
                  fill=color, width=lw)

        value = (d.get("value") or "").strip()
        if value:
            mx = (ax + bx) / 2 + nx * gap
            my = (ay + by) / 2 + ny * gap
            bb = draw.textbbox((0, 0), value, font=font)
            draw.text((mx - (bb[0] + bb[2]) / 2, my - (bb[1] + bb[3]) / 2),
                      value, font=font, fill=color)

    layer = layer.resize((w, h), Image.LANCZOS)
    out = img.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def to_png_bytes(img: Image.Image, optimize: bool = False) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=optimize)
    return buf.getvalue()


def to_jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    # progressive: browsers paint a coarse full frame from the first scan and
    # sharpen as bytes arrive — on slow mobile connections the preview appears
    # in a fraction of the full download time (baseline JPEG reveals top-down).
    img.convert("RGB").save(buf, format="JPEG", quality=quality, progressive=True)
    return buf.getvalue()
