"""Ephemeral, isolated knowledge sessions for the local TrueLine prototype."""

from __future__ import annotations

from datetime import datetime
import uuid

from .errors import ValidationFailed


class ActiveSessionError(ValidationFailed):
    """Raised when a session is missing, expired, or already active."""

    http_status = 409


def _clear_knowledge(conn) -> None:
    """Clear all indexed knowledge rows while preserving schema and migrations."""
    conn.execute("UPDATE knowledge_sources SET current_version_id=NULL")
    conn.execute("DELETE FROM knowledge_chunks")
    conn.execute("DELETE FROM knowledge_source_versions")
    conn.execute("DELETE FROM knowledge_sources")


def start_session(conn, organization_id: str) -> dict:
    """Start the sole active session and reset prior prototype knowledge."""
    with conn.transaction():
        active = conn.execute(
            "SELECT session_id FROM knowledge_sessions LIMIT 1"
        ).fetchone()
        if active:
            raise ActiveSessionError("A session is already active. Resume it or end it first.")

        _clear_knowledge(conn)
        session_id = str(uuid.uuid4())
        session_brand_id = f"{organization_id}__session_{session_id.replace('-', '')}"
        row = conn.execute(
            "INSERT INTO knowledge_sessions (session_id, organization_id, session_brand_id) "
            "VALUES (%s,%s,%s) RETURNING started_at",
            (session_id, organization_id, session_brand_id),
        ).fetchone()
    return {"sessionId": session_id, "startedAt": row["started_at"].isoformat()}


def get_session(conn, session_id: str, organization_id: str) -> dict:
    """Resolve an active session token to its isolated retrieval brand scope."""
    try:
        normalized_id = str(uuid.UUID(session_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ActiveSessionError("Start or resume an active session first.") from exc
    row = conn.execute(
        "SELECT session_id, organization_id, session_brand_id, started_at "
        "FROM knowledge_sessions WHERE session_id=%s AND organization_id=%s",
        (normalized_id, organization_id),
    ).fetchone()
    if not row:
        raise ActiveSessionError("This session has ended. Start a new session.")
    return {
        "sessionId": str(row["session_id"]),
        "organizationId": row["organization_id"],
        "brandId": row["session_brand_id"],
        "startedAt": row["started_at"].isoformat() if isinstance(row["started_at"], datetime) else str(row["started_at"]),
    }


def end_session(conn, session_id: str, organization_id: str) -> dict:
    """Delete all current knowledge and close the active session atomically."""
    session = get_session(conn, session_id, organization_id)
    with conn.transaction():
        # The prototype has one organization and one active user session. Ending
        # it clears the complete knowledge index, not just the visible UI state.
        _clear_knowledge(conn)
        conn.execute("DELETE FROM knowledge_sessions")
    return {"ended": True, "sessionId": session["sessionId"], "cleared": True}


def list_sources(conn, brand_id: str) -> list[dict]:
    """List current indexed sources in one active session."""
    rows = conn.execute(
        "SELECT s.source_id, s.brand_id, s.source_type, s.title, "
        "v.document_type, v.status, v.approval_status, v.version_id, "
        "v.version_number AS version, jsonb_array_length(v.rule_data) AS rule_count, "
        "count(c.chunk_id)::int AS chunk_count "
        "FROM knowledge_sources s "
        "JOIN knowledge_source_versions v ON v.version_id=s.current_version_id "
        "LEFT JOIN knowledge_chunks c ON c.version_id=v.version_id "
        "WHERE s.brand_id=%s AND v.brand_id=%s "
        "GROUP BY s.source_id, s.brand_id, s.source_type, s.title, v.document_type, "
        "v.status, v.approval_status, v.version_id, v.version_number, v.rule_data "
        "ORDER BY s.created_at DESC",
        (brand_id, brand_id),
    ).fetchall()
    return [dict(row) for row in rows]
