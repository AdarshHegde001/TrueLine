"""Deterministic chunking. Every chunk_text is an EXACT slice of the source text,
so text[char_start:char_end] == chunk_text always holds (offsets are into the
canonical `knowledge_source_versions.text`, Python str indices)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    char_start: int
    char_end: int


def _cut_point(text: str, start: int, hard_end: int, min_len: int) -> int:
    """Best break at/before hard_end: paragraph > sentence > whitespace > hard cut."""
    lo = start + min_len
    for sep in ("\n\n", ". ", "! ", "? ", "\n", " "):
        i = text.rfind(sep, lo, hard_end)
        if i != -1:
            return i + len(sep)
    return hard_end


def chunk_text(text: str, max_chars: int = 800) -> list[Chunk]:
    n = len(text)
    chunks: list[Chunk] = []
    pos = 0
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        hard_end = min(pos + max_chars, n)
        end = hard_end if hard_end == n else _cut_point(text, pos, hard_end, max_chars // 4)
        piece_end = end
        while piece_end > pos and text[piece_end - 1].isspace():
            piece_end -= 1
        if piece_end > pos:
            chunks.append(Chunk(len(chunks), text[pos:piece_end], pos, piece_end))
        pos = end
    return chunks
