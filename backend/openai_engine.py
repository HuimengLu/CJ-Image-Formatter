"""OpenAI gpt-image-1 background/shadow generation (Testing 2 path).

Given a product photo, ask the model to redraw it on a solid white background
with a realistic studio shadow. The white
plate is later multiply-blended onto the CJ backdrop (pipeline.compose_testing2),
so no alpha is needed — pure white multiplies to identity and only the shadow
darkens the backdrop.

Uses the image *edit* endpoint (edits the uploaded photo rather than generating
from scratch), with input_fidelity=high to preserve the product's real detail.

Auth: OPENAI_API_KEY from the environment or a repo-root .env. The key value
never appears in code. Note: gpt-image-1 requires a verified OpenAI org.
"""
import base64
import io
import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()
log = logging.getLogger("cj_api")

MODEL = os.environ.get("CJ_OPENAI_MODEL", "gpt-image-2")
SIZE = os.environ.get("CJ_OPENAI_SIZE", "1536x1024")   # 1024x1024 | 1536x1024 | 1024x1536 | auto
QUALITY = os.environ.get("CJ_OPENAI_QUALITY", "low")   # low | medium | high | auto
MODERATION = os.environ.get("CJ_OPENAI_MODERATION", "low")  # low | auto
# Empty → don't send input_fidelity; API applies its own default (low). Set to
# "high" to force the model to preserve fine product detail.
FIDELITY = os.environ.get("CJ_OPENAI_FIDELITY", "")

# Retry/backoff — image models are rate-limited per minute; on a 429 wait most
# of a minute rather than failing the upload.
RETRIES = int(os.environ.get("CJ_OPENAI_RETRIES", "3"))
BACKOFF = float(os.environ.get("CJ_OPENAI_BACKOFF", "6"))       # base seconds
RATE_WAIT = float(os.environ.get("CJ_OPENAI_RATE_WAIT", "30"))  # extra on 429

# Longest side sent to the API. Outputs cap at ~1536px, so full-resolution
# uploads (a 13MP phone photo is a 15-25MB PNG) only add upload/preprocess
# time — they measurably slowed generations from ~15s to ~50s.
MAX_SEND = int(os.environ.get("CJ_OPENAI_MAX_SEND", "2048"))


def _png_buf(pil_img: Image.Image, name: str) -> io.BytesIO:
    """PNG-encode an image for upload, downscaled to MAX_SEND on the long side."""
    img = pil_img.convert("RGB")
    if max(img.size) > MAX_SEND:
        img = img.copy()
        img.thumbnail((MAX_SEND, MAX_SEND), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.name = name
    return buf

# The fixed prompt: remove background → solid white + front-right studio shadow,
# keep the product's real texture/detail (no cut-out look).
PROMPT = (
    "Remove the image background and replace it with a solid white backdrop. "
    "Preserve the product exactly as captured without changing its geometry, "
    "color, texture, or perspective. Add one soft natural floor shadow that "
    "originates from the object's contact points and extends diagonally toward "
    "the back-left of the image, matching illumination from the front-right. "
    "The shadow should be subtle, short, and softly blurred, fading naturally "
    "with distance. Do not place the shadow directly underneath the object or "
    "evenly around it. Scale the product so that its longest dimension occupies "
    "approximately 75% of the frame. Maintain consistent framing and "
    "generous white space around the object across all images."
)

# ── Cover mode: category classification + scene composition ────────────────────

CLASSIFY_MODEL = os.environ.get("CJ_OPENAI_CLASSIFY_MODEL", "gpt-5.4")

# GPT-5 and o-series reason before answering and bill those hidden tokens
# against the completion budget, so the 4.1-era params (max_tokens=20,
# temperature=0) would come back empty — and a non-default temperature is
# rejected outright. Pick the param set from the model family so overriding
# CJ_OPENAI_CLASSIFY_MODEL back to a 4.x model still works.
_CLASSIFY_REASONS = not CLASSIFY_MODEL.startswith(("gpt-4", "gpt-3"))
CLASSIFY_PARAMS = ({"max_completion_tokens": 512} if _CLASSIFY_REASONS
                   else {"max_tokens": 20, "temperature": 0})

# The five cover categories; classification must return exactly one of these.
CATEGORIES = [
    "Furniture",
    "Appliances & Fixtures",
    "Building Materials/Outdoor",
    "Lighting",
    "Specialty",
]

CLASSIFY_PROMPT = (
    "Identify the primary product shown in the image. Then answer with exactly "
    "one of the following category names and nothing else. Choose by the "
    "descriptions, not by the category names themselves:\n"
    "Furniture — ALL ordinary indoor home, office and commercial furniture: "
    "chairs, tables, desks, sofas, beds, dressers, shelving, individual "
    "cabinets and other cabinetry. Patio, garden and other outdoor furniture "
    "does NOT belong here — put it in Building Materials/Outdoor\n"
    "Appliances & Fixtures — ONLY appliances (refrigerator, stove, washer, "
    "etc.), HVAC equipment, plumbing and bath fixtures (sink, tub, toilet), "
    "and complete kitchen cabinet sets / casework. Never use this category for "
    "ordinary furniture such as chairs or tables\n"
    "Building Materials/Outdoor — building materials and lumber, countertops, "
    "doors, garden and outdoor items, windows, shutters and skylights\n"
    "Lighting — lamps, light fixtures and ceiling fans\n"
    "Specialty — special or uncommon items that fit none of the above"
)

# One prompt per category, each describing its actual backdrop image — a
# mismatched description makes the model invent the missing elements. All
# five share the same philosophy: place the product INTO the provided
# backdrop and never repaint it (a "style reference / recreate" phrasing
# regenerates the scene every time, so backgrounds drift between listings).
# Sizing is visual-consistency (product ≈ 50% of image height) rather than
# true physical scale, so "scale" is deliberately absent from the
# preserve-exactly lists. Seam positions are "as provided", not numeric —
# quoting a percentage invites the model to move the seam to match it.

COVER_PROMPT_FURNITURE = (
    "Remove the background from the product image and place the product into "
    "the provided industrial studio backdrop. Preserve the product exactly as "
    "captured without changing its geometry, perspective, color, texture, "
    "materials, or proportions. Treat the plain white plaster wall, dark "
    "baseboard, and smooth concrete floor as a permanent studio backdrop. "
    "Never modify the background geometry, perspective, lighting, or "
    "proportions. Keep the wall-floor seam exactly where it appears in the "
    "provided backdrop; never move it. Use a fixed straight-on camera with a "
    "constant focal length. Keep the camera height, viewing angle, framing, "
    "and perspective identical across every generation. Scale and position "
    "the product so it occupies approximately 50% of the image height, "
    "centered horizontally and resting naturally on the floor. Illuminate "
    "the product using one large soft studio light positioned at the "
    "front-right (approximately 45° horizontally and 35° above the object). "
    "Keep the lighting identical in every image. Blend the product naturally "
    "into the environment by matching the scene's perspective, lighting "
    "direction, color temperature, and floor plane. Generate one realistic "
    "contact shadow extending diagonally toward the back-left, originating "
    "from the object's contact points. The shadow should be soft, short, and "
    "physically accurate. The final result should look like a professionally "
    "photographed product placed in the same industrial studio set, not a "
    "composited cutout."
)

COVER_PROMPT_APPLIANCES = (
    "Remove the background from the product image and place the product into "
    "the provided industrial warehouse corner. Preserve the product exactly "
    "as captured without changing its geometry, perspective, color, texture, "
    "materials, or proportions. Treat the warehouse corner — the red brick "
    "wall with its large factory window on the left, the white plaster wall "
    "on the right, the wooden ceiling beams, and the aged concrete floor — "
    "as a permanent backdrop. Never modify the background geometry, "
    "perspective, lighting, or proportions. Keep the wall-floor boundary "
    "exactly where it appears in the provided backdrop; never move it. Use a "
    "fixed camera with a constant focal length. Keep the camera height, "
    "viewing angle, framing, and perspective identical across every "
    "generation. Scale and position the product so it occupies approximately "
    "50% of the image height and rests naturally on the floor. Illuminate "
    "the product with the soft natural daylight coming through the factory "
    "window on the left. Keep the lighting identical in every image. Blend "
    "the product naturally into the environment by matching the scene's "
    "perspective, lighting direction, color temperature, and floor plane. "
    "Generate one realistic contact shadow consistent with the window light, "
    "extending gently away from the window, originating from the object's "
    "contact points. The shadow should be soft, short, and physically "
    "accurate. The final result should look like a professionally "
    "photographed product placed in the same warehouse corner, not a "
    "composited cutout."
)

COVER_PROMPT_OUTDOOR = (
    "Remove the background from the product image and place the product into "
    "the provided outdoor backdrop. Preserve the product exactly as captured "
    "without changing its geometry, perspective, color, texture, materials, "
    "or proportions. Treat the sunlit red brick wall with its dappled tree "
    "shadows and the herringbone brick-paved ground as a permanent backdrop. "
    "Never modify the background geometry, perspective, lighting, or "
    "proportions. Keep the wall-ground seam exactly where it appears in the "
    "provided backdrop; never move it. Use a fixed straight-on camera with a "
    "constant focal length. Keep the camera height, viewing angle, framing, "
    "and perspective identical across every generation. Scale and position "
    "the product so it occupies approximately 50% of the image height, "
    "centered horizontally and resting naturally on the ground. Illuminate "
    "the product with the scene's warm natural daylight. Keep the lighting "
    "identical in every image. Blend the product naturally into the "
    "environment by matching the scene's perspective, lighting direction, "
    "color temperature, and ground plane. Generate one realistic contact "
    "shadow consistent with the scene's daylight, originating from the "
    "object's contact points. The shadow should be soft, short, and "
    "physically accurate. The final result should look like a professionally "
    "photographed product placed in the same outdoor set, not a composited "
    "cutout."
)

COVER_PROMPT_SPECIALTY = (
    "Remove the background from the product image and place the product into "
    "the provided industrial studio backdrop. Preserve the product exactly as "
    "captured without changing its geometry, perspective, color, texture, "
    "materials, or proportions. Treat the red brick wall, dark baseboard, "
    "and smooth concrete floor as a permanent studio backdrop. Never modify "
    "the background geometry, perspective, lighting, or proportions. Keep "
    "the wall-floor seam exactly where it appears in the provided backdrop; "
    "never move it. Use a fixed straight-on camera with a constant focal "
    "length. Keep the camera height, viewing angle, framing, and perspective "
    "identical across every generation. Scale and position the product so it "
    "occupies approximately 50% of the image height, centered horizontally "
    "and resting naturally on the floor. Illuminate the product using one "
    "large soft studio light positioned at the front-right (approximately "
    "45° horizontally and 35° above the object). Keep the lighting identical "
    "in every image. Blend the product naturally into the environment by "
    "matching the scene's perspective, lighting direction, color "
    "temperature, and floor plane. Generate one realistic contact shadow "
    "extending diagonally toward the back-left, originating from the "
    "object's contact points. The shadow should be soft, short, and "
    "physically accurate. The final result should look like a professionally "
    "photographed product placed in the same industrial studio set, not a "
    "composited cutout."
)

COVER_PROMPT_LIGHTING = (
    "Replace the background with the designated lighting scene while "
    "preserving the lamp exactly as captured. Do not alter its geometry, "
    "proportions, perspective, color, texture, finish, materials, or any "
    "visible details. If the lamp is illuminated in the original image, "
    "preserve its original light emission, brightness, glow, and color "
    "temperature exactly as captured. Do not add, remove, or exaggerate the "
    "lighting effect. If the lamp is turned off, keep it turned off. Match "
    "the scene lighting naturally with the lamp while maintaining realistic "
    "reflections on glass, metal, and glossy surfaces. Add soft contact "
    "shadows that are consistent with the scene lighting without changing the "
    "lamp itself. Center the lamp within the composition and scale it "
    "naturally so its longest dimension occupies approximately 60–65% of the "
    "frame, leaving balanced negative space around all sides. Maintain a "
    "consistent camera distance, framing, and composition across all "
    "generated images. The final result should resemble a professional "
    "lifestyle product photograph, with the lamp remaining the visual focus "
    "while integrating naturally into the environment."
)

# Category → prompt. Keys must stay in sync with CATEGORIES; Specialty is the
# fallback for anything unmapped, mirroring classify_category's fallback.
COVER_PROMPTS = {
    "Furniture": COVER_PROMPT_FURNITURE,
    "Appliances & Fixtures": COVER_PROMPT_APPLIANCES,
    "Building Materials/Outdoor": COVER_PROMPT_OUTDOOR,
    "Lighting": COVER_PROMPT_LIGHTING,
    "Specialty": COVER_PROMPT_SPECIALTY,
}

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazily build the client so a missing key only errors on first use."""
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set — put it in the environment or a "
                "repo-root .env file (OPENAI_API_KEY=...)."
            )
        _client = OpenAI(api_key=key)
    return _client


def _humanize(raw: str) -> str:
    """Translate raw API/SDK errors into staff-facing messages.

    The raw error is already in the server log (each retry attempt logs it);
    the UI should tell a non-engineer what to do next, not show a stack dump.
    """
    m = raw.lower()
    if "429" in raw or "rate" in m or "quota" in m or "insufficient" in m or "exhausted" in m:
        return "Processing quota is used up — wait a minute and retry, or contact the admin."
    if "401" in raw or "api key" in m or "auth" in m:
        return "API key problem — contact the admin."
    if "moderation" in m or "safety" in m or "content policy" in m or "rejected" in m:
        return "The image was declined by the processor — try a different photo."
    if "timeout" in m or "timed out" in m or "connect" in m or "network" in m or "unavailable" in m:
        return "Network hiccup while processing — please retry this photo."
    return "Processing failed — please retry this photo."


def _decode(resp) -> Image.Image | None:
    """Pull the first image out of an images.edit response (always b64 here)."""
    for item in getattr(resp, "data", None) or []:
        b64 = getattr(item, "b64_json", None)
        if b64:
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return None


def white_bg_with_shadow(pil_img: Image.Image) -> tuple[Image.Image | None, str | None]:
    """Send one image to gpt-image-1; return (white_bg_rgb, warn).

    On success warn is None; on failure the image is None and warn holds a
    short message. Retries with backoff cover transient / rate-limit hiccups.
    """
    src = _png_buf(pil_img, "input.png")

    last = "OpenAI returned no image"
    for attempt in range(RETRIES):
        try:
            src.seek(0)
            # moderation / input_fidelity aren't named params on every SDK
            # version; extra_body forwards them as form fields regardless.
            extra = {}
            if MODERATION:
                extra["moderation"] = MODERATION
            if FIDELITY:
                extra["input_fidelity"] = FIDELITY
            resp = _get_client().images.edit(
                model=MODEL, image=src, prompt=PROMPT,
                size=SIZE, quality=QUALITY, n=1,
                extra_body=extra or None)
            img = _decode(resp)
            if img is not None:
                return img, None
        except Exception as exc:             # network / auth / quota / rate limit
            last = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("OpenAI call failed (attempt %d/%d): %s",
                        attempt + 1, RETRIES, last)
        if attempt < RETRIES - 1:
            rate_limited = "429" in last or "rate" in last.lower()
            wait = BACKOFF * (attempt + 1) + (RATE_WAIT if rate_limited else 0)
            log.info("OpenAI retry in %.0fs", wait)
            time.sleep(wait)
    return None, _humanize(last)


def classify_category(pil_img: Image.Image) -> str:
    """Steps 1+2 of Cover mode in one CLASSIFY_MODEL vision call.

    Identifies the product and maps it to one of CATEGORIES; any failure
    (network, refusal, unparseable answer) falls back to "Specialty" so the
    cover flow always proceeds.
    """
    small = pil_img.convert("RGB").copy()
    small.thumbnail((768, 768))              # plenty for classification, fewer tokens
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        resp = _get_client().chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": CLASSIFY_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            **CLASSIFY_PARAMS)
        answer = (resp.choices[0].message.content or "").strip().lower()
    except Exception as exc:
        log.warning("classify_category failed (%s) — using Specialty", exc)
        return "Specialty"

    if not answer:
        # Systematic, not per-image: the model spent its whole budget on
        # reasoning tokens. Raise CLASSIFY_PARAMS rather than ignoring this —
        # otherwise every cover silently lands on Specialty.
        log.error("classify_category: empty answer from %s — using Specialty; "
                  "completion budget is likely too small", CLASSIFY_MODEL)
        return "Specialty"

    answer = answer.strip(" .\"'*")

    # Exact match first: "Furniture" is a substring of chatty answers about
    # other categories ("appliance, not furniture"), so a plain containment
    # check on a first-listed category would mis-bucket them.
    for cat in CATEGORIES:
        if cat.lower() == answer:
            log.info("classify_category: %r -> %s", answer[:60], cat)
            return cat
    # Then both containment directions: the answer may wrap the name
    # ("Category: Lighting"), or abbreviate a multi-word one — "Appliances"
    # for "Appliances & Fixtures" — which the wrapping check alone would miss.
    for cat in CATEGORIES:
        low = cat.lower()
        if low in answer or answer in low:
            log.info("classify_category: %r -> %s (loose)", answer[:60], cat)
            return cat
    log.warning("classify_category: unrecognised answer %r — using Specialty",
                answer[:60])
    return "Specialty"


def cover_scene(product_img: Image.Image, bg_img: Image.Image,
                category: str) -> tuple[Image.Image | None, str | None]:
    """Step 3 of Cover mode: compose the product into the category scene.

    Sends [product, background] to gpt-image-2 — product first so it is the
    edit subject, background second as the scene to place it into — with the
    category's own prompt from COVER_PROMPTS. Returns (scene_rgb, warn); same
    retry/backoff behaviour as white_bg_with_shadow.
    """
    prompt = COVER_PROMPTS.get(category, COVER_PROMPT_SPECIALTY)

    bufs = [_png_buf(product_img, "product.png"),
            _png_buf(bg_img, "background.png")]

    last = "OpenAI returned no image"
    for attempt in range(RETRIES):
        try:
            for b in bufs:
                b.seek(0)
            extra = {}
            if MODERATION:
                extra["moderation"] = MODERATION
            if FIDELITY:
                extra["input_fidelity"] = FIDELITY
            resp = _get_client().images.edit(
                model=MODEL, image=bufs, prompt=prompt,
                size=SIZE, quality=QUALITY, n=1,
                extra_body=extra or None)
            img = _decode(resp)
            if img is not None:
                return img, None
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("cover_scene failed (attempt %d/%d): %s",
                        attempt + 1, RETRIES, last)
        if attempt < RETRIES - 1:
            rate_limited = "429" in last or "rate" in last.lower()
            wait = BACKOFF * (attempt + 1) + (RATE_WAIT if rate_limited else 0)
            log.info("OpenAI retry in %.0fs", wait)
            time.sleep(wait)
    return None, _humanize(last)
