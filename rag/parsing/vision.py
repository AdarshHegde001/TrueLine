"""Small local image features used to screen logo candidates before any model call."""

from __future__ import annotations


def image_dhash(data: bytes) -> str:
    """Return a 64-bit difference hash; no image bytes leave the local process."""
    import fitz

    source = fitz.Pixmap(data)
    if source.width < 2 or source.height < 1:
        raise ValueError("image is too small for visual hashing")
    gray = fitz.Pixmap(fitz.csGRAY, source)
    samples = gray.samples
    width, height = gray.width, gray.height
    bits = 0
    for y in range(8):
        for x in range(8):
            row = min(height - 1, (y * height) // 8)
            left = min(width - 1, (x * width) // 9)
            right = min(width - 1, ((x + 1) * width) // 9)
            bits = (bits << 1) | int(samples[row * width + left] > samples[row * width + right])
    return f"{bits:016x}"


def normalized_image_text(text: str) -> set[str]:
    """Normalize OCR text to meaningful tokens for local logo-name matching."""
    import re

    return {
        token for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) >= 3 and not token.isdigit()
    }


def hash_similarity(left: str, right: str) -> float:
    try:
        a, b = int(left, 16), int(right, 16)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 - ((a ^ b).bit_count() / 64.0)
