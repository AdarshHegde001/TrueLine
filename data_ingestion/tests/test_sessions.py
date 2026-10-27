"""Tests for the session lifecycle: start, resume, end, and stale-session recovery.

Needs a Postgres instance: set TEST_DATABASE_URL (tests DROP + recreate the session table)."""
import os

import psycopg
import pytest
from psycopg.rows import dict_row

from trueline_ingestion import db
from trueline_ingestion.config import get_settings
from trueline_ingestion.sessions import (
    ActiveSessionError,
    start_session,
    get_session,
    end_session,
)

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")

S = get_settings()


@pytest.fixture()
def conn():
    c = psycopg.connect(URL, row_factory=dict_row)
    # Clean slate: drop all knowledge + session tables and re-migrate.
    c.execute(
        "DROP TABLE IF EXISTS knowledge_chunks, knowledge_source_versions, "
        "knowledge_sources, knowledge_sessions CASCADE"
    )
    c.commit()
    db.migrate(c)
    yield c
    c.close()


ORG = "brand_test"


# ---- basic lifecycle -----------------------------------------------------------

def test_start_session_returns_id_and_timestamp(conn):
    result = start_session(conn, ORG)
    assert "sessionId" in result and "startedAt" in result
    assert len(result["sessionId"]) == 36  # UUID format


def test_resume_session_returns_brand_id(conn):
    created = start_session(conn, ORG)
    resumed = get_session(conn, created["sessionId"], ORG)
    assert resumed["sessionId"] == created["sessionId"]
    assert resumed["brandId"].startswith(ORG)
    assert resumed["organizationId"] == ORG


def test_end_session_clears_session(conn):
    created = start_session(conn, ORG)
    result = end_session(conn, created["sessionId"], ORG)
    assert result["ended"] is True
    # Trying to resume should now fail
    with pytest.raises(ActiveSessionError):
        get_session(conn, created["sessionId"], ORG)


# ---- stale session auto-clear (our fix) -----------------------------------------

def test_start_session_auto_clears_stale_session(conn):
    """Starting a new session when a stale one exists should succeed, not raise."""
    first = start_session(conn, ORG)
    # Don't end the first session — simulate a crash / unclean shutdown.
    # Starting another should auto-clear the stale one and succeed.
    second = start_session(conn, ORG)
    assert second["sessionId"] != first["sessionId"]
    # The old session should be gone.
    with pytest.raises(ActiveSessionError):
        get_session(conn, first["sessionId"], ORG)
    # The new session should be active.
    resumed = get_session(conn, second["sessionId"], ORG)
    assert resumed["sessionId"] == second["sessionId"]


def test_start_after_clean_end_works(conn):
    """The normal flow: start → end → start again should work."""
    first = start_session(conn, ORG)
    end_session(conn, first["sessionId"], ORG)
    second = start_session(conn, ORG)
    assert second["sessionId"] != first["sessionId"]


# ---- error cases ---------------------------------------------------------------

def test_resume_invalid_session_id_raises(conn):
    with pytest.raises(ActiveSessionError):
        get_session(conn, "not-a-uuid", ORG)


def test_resume_nonexistent_session_raises(conn):
    with pytest.raises(ActiveSessionError):
        get_session(conn, "00000000-0000-0000-0000-000000000000", ORG)


def test_end_nonexistent_session_raises(conn):
    with pytest.raises(ActiveSessionError):
        end_session(conn, "00000000-0000-0000-0000-000000000000", ORG)


def test_resume_with_wrong_org_raises(conn):
    created = start_session(conn, ORG)
    with pytest.raises(ActiveSessionError):
        get_session(conn, created["sessionId"], "wrong_org")
