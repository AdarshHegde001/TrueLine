"""Session-scoped storage for approved logo reference images."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePath
import shutil
import uuid

from .errors import ValidationFailed, NotFound

SUPPORTED_LOGO_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}
MAX_LOGOS_PER_SESSION = 12
MAX_LOGO_BYTES = 5 * 1024 * 1024


def asset_root() -> Path:
    default = Path(__file__).resolve().parents[1] / "extracted_assets"
    return Path(os.environ.get("TRUELINE_ASSET_DIR", default)).resolve()


def _safe_asset_path(storage_ref: str) -> Path:
    root = asset_root()
    candidate = (root / storage_ref).resolve()
    if root not in candidate.parents:
        raise ValidationFailed("The stored logo reference is invalid")
    return candidate


def add_logo(conn, *, brand_id: str, title: str, filename: str,
             content_type: str, data: bytes) -> dict:
    extension = SUPPORTED_LOGO_TYPES.get(content_type.lower())
    if extension is None:
        raise ValidationFailed("Logo files must be PNG, JPEG, or WebP images")
    if not data or len(data) > MAX_LOGO_BYTES:
        raise ValidationFailed("Logo files must be smaller than 5 MB")

    try:
        import fitz
        with fitz.open(stream=data, filetype=extension) as image_document:
            if image_document.page_count != 1:
                raise ValueError("image page count")
            rect = image_document[0].rect
            width, height = int(rect.width), int(rect.height)
            if width < 1 or height < 1 or width * height > 100_000_000:
                raise ValueError("image dimensions")
    except Exception as exc:
        raise ValidationFailed("The uploaded logo is not a readable image") from exc

    digest = hashlib.sha256(data).hexdigest()
    from rag.parsing.image import ImageParser
    from rag.parsing.vision import image_dhash

    extracted_text = ImageParser().parse(
        data, document_name=f"logo.{extension}", store_images=False,
    ).text[:1000]
    visual_hash = image_dhash(data)
    duplicate = conn.execute(
        "SELECT logo_id, title, original_name, content_type, width, height, created_at "
        "FROM knowledge_logos WHERE brand_id=%s AND sha256=%s",
        (brand_id, digest),
    ).fetchone()
    if duplicate:
        return _public_logo(duplicate)

    count = conn.execute(
        "SELECT count(*) AS n FROM knowledge_logos WHERE brand_id=%s", (brand_id,)
    ).fetchone()["n"]
    if count >= MAX_LOGOS_PER_SESSION:
        raise ValidationFailed(f"A session can store up to {MAX_LOGOS_PER_SESSION} logo references")

    logo_id = f"logo_{uuid.uuid4().hex}"
    safe_name = PurePath(filename.replace("\\", "/")).name or f"{logo_id}.{extension}"
    root = asset_root()
    directory = root / logo_id
    storage_ref = f"{logo_id}/{digest}.{extension}"
    try:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / f"{digest}.{extension}").write_bytes(data)
        row = conn.execute(
            "INSERT INTO knowledge_logos "
            "(logo_id, brand_id, title, original_name, content_type, storage_ref, sha256, "
            "visual_hash, ocr_text, width, height) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (brand_id, sha256) DO NOTHING "
            "RETURNING logo_id, title, original_name, content_type, width, height, created_at",
            (logo_id, brand_id, title.strip() or Path(safe_name).stem, safe_name,
             content_type.lower(), storage_ref, digest, visual_hash, extracted_text, width, height),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT logo_id, title, original_name, content_type, width, height, created_at "
                "FROM knowledge_logos WHERE brand_id=%s AND sha256=%s",
                (brand_id, digest),
            ).fetchone()
            shutil.rmtree(directory, ignore_errors=True)
        return _public_logo(row)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def list_logos(conn, brand_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT logo_id, title, original_name, content_type, width, height, created_at "
        "FROM knowledge_logos WHERE brand_id=%s ORDER BY created_at, logo_id",
        (brand_id,),
    ).fetchall()
    return [_public_logo(row) for row in rows]


def get_logo_assets(conn, brand_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT logo_id, title, original_name, content_type, storage_ref, sha256, "
        "visual_hash, ocr_text, width, height "
        "FROM knowledge_logos WHERE brand_id=%s ORDER BY created_at, logo_id",
        (brand_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def read_logo(conn, brand_id: str, logo_id: str) -> tuple[dict, bytes]:
    row = conn.execute(
        "SELECT logo_id, content_type, storage_ref FROM knowledge_logos "
        "WHERE brand_id=%s AND logo_id=%s", (brand_id, logo_id)
    ).fetchone()
    if row is None:
        raise NotFound("logo not found")
    path = _safe_asset_path(row["storage_ref"])
    try:
        return dict(row), path.read_bytes()
    except OSError as exc:
        raise NotFound("stored logo file is unavailable") from exc


def read_storage_ref(storage_ref: str) -> bytes:
    path = _safe_asset_path(storage_ref)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise NotFound("stored image file is unavailable") from exc


def delete_logo(conn, brand_id: str, logo_id: str) -> dict:
    with conn.transaction():
        row = conn.execute(
            "DELETE FROM knowledge_logos WHERE brand_id=%s AND logo_id=%s "
            "RETURNING storage_ref", (brand_id, logo_id)
        ).fetchone()
        if row is None:
            raise NotFound("logo not found")
    path = _safe_asset_path(row["storage_ref"])
    shutil.rmtree(path.parent, ignore_errors=True)
    return {"deleted": True, "logoId": logo_id}


def _public_logo(row) -> dict:
    return {
        "logo_id": row["logo_id"],
        "title": row["title"],
        "original_name": row["original_name"],
        "content_type": row["content_type"],
        "width": row["width"],
        "height": row["height"],
        "created_at": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
    }
