#!/usr/bin/env python3
"""
Gemini Image Generation Tool

Generates images from text prompts using Google's Gemini native image models
(aka "nano-banana"). Preferred over FAL FLUX 2 Pro when the image needs to
render *legible text* (logos, posters, magazine covers, infographics, memes).

Saves images to ~/.hermes/images/<uuid>.png and returns a JSON result with
file paths that chat frontends can render via markdown.

Environment:
    GOOGLE_API_KEY: required (Gemini API key)

Models (priority order, auto-fallback if a model is unavailable):
    1. gemini-3.1-flash-image-preview  (latest, fast, recommended)
    2. gemini-3-pro-image-preview      (highest quality)
    3. gemini-2.5-flash-image          (stable fallback)
"""

import base64
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
FALLBACK_MODELS = [
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
]

IMAGES_DIR = Path.home() / ".hermes" / "images"

ASPECT_HINT = {
    "landscape": "16:9 landscape composition, widescreen",
    "square": "1:1 square composition, centered",
    "portrait": "9:16 portrait composition, vertical",
}

TEXT_STYLE_TEMPLATES = {
    "magazine": (
        "Rendered as a glossy magazine cover layout. Headline typography in "
        "a bold serif font, large and filling the frame. Clean high-contrast "
        "composition, editorial design."
    ),
    "infographic": (
        "Rendered as a vibrant infographic / educational poster. Clean sans-serif "
        "labels, well-spaced, crisp and legible. Colorful but organized layout."
    ),
    "poster": (
        "Rendered as a bold minimal poster. High-contrast typography, centered "
        "composition, strong visual hierarchy, clean sans-serif."
    ),
    "logo": (
        "Rendered as a clean vector logo design. Symmetric, balanced, brand-mark "
        "quality. Crisp letterforms, no background clutter."
    ),
    "meme": (
        "Rendered as a classic meme layout with Impact font text, white fill and "
        "black outline, top and/or bottom placement."
    ),
}

TEXT_RENDER_COACH = (
    "Render ALL text shown in quotes as crisp, correctly-spelled, legible "
    "typography. Each letter must be sharp and readable against the background. "
    "High contrast. No garbled characters, no duplicated letters."
)


def _detect_quoted_text(prompt: str) -> List[str]:
    """Return any substrings wrapped in ASCII or fullwidth quotes."""
    patterns = [
        r'"([^"]+)"',
        r"'([^']+)'",
        r'「([^」]+)」',
        r'『([^』]+)』',
        r'“([^”]+)”',
    ]
    found: List[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, prompt))
    return found


def _build_final_prompt(
    user_prompt: str, aspect_ratio: str, text_style: Optional[str]
) -> str:
    """Apply the text-rendering prompt wrapper when appropriate.

    Strategy:
      * Always add an aspect-ratio hint (Gemini does not accept size as a param).
      * If ``text_style`` is given, append the matching style template.
      * If quoted text is detected and no explicit style, append the generic coach.
    """
    parts = [user_prompt.strip()]
    parts.append(ASPECT_HINT.get(aspect_ratio, ASPECT_HINT["landscape"]))

    if text_style and text_style in TEXT_STYLE_TEMPLATES:
        parts.append(TEXT_STYLE_TEMPLATES[text_style])
        parts.append(TEXT_RENDER_COACH)
    elif _detect_quoted_text(user_prompt):
        parts.append(TEXT_RENDER_COACH)

    return " ".join(parts)


def _save_image_bytes(data: bytes, mime: str) -> Path:
    ext = "png"
    if mime and "/" in mime:
        ext = mime.split("/", 1)[1].lower()
        if ext == "jpeg":
            ext = "jpg"
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGES_DIR / f"gemini_{uuid.uuid4().hex[:12]}.{ext}"
    path.write_bytes(data)
    return path


def _try_generate(client, model: str, final_prompt: str) -> List[Path]:
    """Call Gemini generate_content and return saved file paths."""
    resp = client.models.generate_content(model=model, contents=final_prompt)
    saved: List[Path] = []

    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            mime = getattr(inline, "mime_type", "image/png") or "image/png"
            if data is None:
                continue
            if isinstance(data, str):
                data = base64.b64decode(data)
            saved.append(_save_image_bytes(data, mime))
    return saved


def gemini_image_tool(
    prompt: str,
    aspect_ratio: str = "landscape",
    text_style: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Synchronous entry point used by the registry handler."""
    if not prompt or not prompt.strip():
        return tool_error("prompt is required for image generation")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return tool_error(
            "GOOGLE_API_KEY is not set. Add it to ~/.hermes/hermes-agent/.env"
        )

    try:
        from google import genai
    except ImportError:
        return tool_error(
            "google-genai not installed. Run: "
            "uv pip install --python <venv-python> google-genai"
        )

    client = genai.Client(api_key=api_key)
    final_prompt = _build_final_prompt(prompt, aspect_ratio, text_style)

    models_to_try: List[str] = []
    if model:
        models_to_try.append(model)
    for m in FALLBACK_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    last_error: Optional[str] = None
    for candidate_model in models_to_try:
        try:
            saved = _try_generate(client, candidate_model, final_prompt)
        except Exception as e:
            last_error = f"{candidate_model}: {type(e).__name__}: {str(e)[:300]}"
            logger.warning("gemini_image_generate failed on %s: %s", candidate_model, e)
            continue
        if not saved:
            last_error = f"{candidate_model}: no image data in response"
            continue

        paths = [str(p) for p in saved]
        markdown = "\n".join(f"![generated]({'file://' + p})" for p in paths)
        result: Dict[str, Any] = {
            "status": "ok",
            "model": candidate_model,
            "aspect_ratio": aspect_ratio,
            "text_style": text_style,
            "final_prompt": final_prompt,
            "num_images": len(paths),
            "images": paths,
            "markdown": markdown,
        }
        return json.dumps(result, indent=2, ensure_ascii=False)

    return tool_error(
        f"Gemini image generation failed on all models. Last error: {last_error}"
    )


def check_gemini_image_requirements() -> bool:
    if not os.getenv("GOOGLE_API_KEY"):
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GEMINI_IMAGE_GENERATE_SCHEMA = {
    "name": "image_generate_gemini",
    "description": (
        "Generate images from a text prompt using Google's Gemini (nano-banana) "
        "native image model. PREFERRED over image_generate when the image must "
        "contain *legible, correctly-spelled text* (logos, posters, magazine "
        "covers, infographics, memes, diagrams, marketing assets). Put any "
        "exact text you want rendered inside quotes. Optionally pass "
        "text_style for composition hints. Saves PNG to ~/.hermes/images/ and "
        "returns a JSON blob with the file path. Display it using markdown: "
        "![description](file:///path/to/image.png)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Narrative scene description. Put any in-image text in quotes, "
                    "e.g. the large bold words \"Hello World\". Describe scenes "
                    "rather than listing keywords."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["landscape", "square", "portrait"],
                "description": "landscape=16:9, portrait=9:16, square=1:1.",
                "default": "landscape",
            },
            "text_style": {
                "type": "string",
                "enum": ["magazine", "infographic", "poster", "logo", "meme"],
                "description": (
                    "Optional composition preset for text-heavy images. Omit for "
                    "general scenes; auto-applies text-rendering coach when quoted "
                    "text is detected in the prompt."
                ),
            },
        },
        "required": ["prompt"],
    },
}


def _handle_gemini_image_generate(args, **kw):
    return gemini_image_tool(
        prompt=args.get("prompt", ""),
        aspect_ratio=args.get("aspect_ratio", "landscape"),
        text_style=args.get("text_style"),
        model=args.get("model"),
    )


registry.register(
    name="image_generate_gemini",
    toolset="image_gen",
    schema=GEMINI_IMAGE_GENERATE_SCHEMA,
    handler=_handle_gemini_image_generate,
    check_fn=check_gemini_image_requirements,
    requires_env=["GOOGLE_API_KEY"],
    is_async=False,
    emoji="🖼️",
)
