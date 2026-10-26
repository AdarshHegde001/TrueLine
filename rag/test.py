"""Manual Gemini API smoke call using a synthetic prompt only.

Run from the repository root after installing rag/requirements.txt and setting
GEMINI_API_KEY. This does not send submitted documents or brand knowledge.
"""

from __future__ import annotations

import os


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY to run this Gemini smoke call.")

    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(
            "Gemini SDK is missing; install dependencies from rag/requirements.txt."
        ) from exc

    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: Gemini API connection works.",
    )
    if not response.text:
        raise SystemExit("Gemini returned no text response.")
    print(response.text)


if __name__ == "__main__":
    main()
