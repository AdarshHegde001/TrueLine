"""Acceptance tests for the ingestion<->RAG contract.

Needs a Postgres with pgvector: set TEST_DATABASE_URL (tests DROP the knowledge_* tables).
Uses FakeEmbedder (hash bag-of-words) so tests do not download BGE-M3 -- these tests verify
plumbing, scoping, versioning and failure behaviour, NOT semantic retrieval quality."""
import os

import psycopg
import pytest
from pydantic import ValidationError
from psycopg.rows import dict_row

from trueline_ingestion import db, ingest
from trueline_ingestion.chunking import chunk_text
from trueline_ingestion.config import get_settings
from trueline_ingestion.embeddings import FakeEmbedder
from trueline_ingestion.errors import IndexingError, NotFound, SearchServiceError
from trueline_ingestion.models import (KnowledgeSearchRequest, SourceCreate, SourceUpdate)
from trueline_ingestion.search import KnowledgeService

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")

S = get_settings()
EMB = FakeEmbedder()


@pytest.fixture()
def conn():
    c = psycopg.connect(URL, row_factory=dict_row)
    c.execute("DROP TABLE IF EXISTS knowledge_chunks, knowledge_source_versions, knowledge_sources CASCADE")
    c.commit()
    db.migrate(c)
    yield c
    c.close()


@pytest.fixture()
def svc():
    return KnowledgeService(EMB, S, conn_factory=lambda: psycopg.connect(URL, row_factory=dict_row))


def add(conn, brand, title, text, stype="guideline", approved=True, **kw):
    return ingest.create_source(
        conn, EMB, SourceCreate(brand_id=brand, source_type=stype, title=title, text=text,
                                approved=approved, **kw), S)


def q(svc, brand, query, **kw):
    return svc.search(KnowledgeSearchRequest(brand_id=brand, query=query, **kw))


# ---- chunking -------------------------------------------------------------------------
def test_chunks_are_exact_slices_of_source():
    text = ("First sentence here. " * 60) + "\n\n" + ("Another paragraph. " * 60)
    chunks = chunk_text(text, 300)
    assert len(chunks) > 3
    for c in chunks:
        assert text[c.char_start:c.char_end] == c.text
        assert len(c.text) <= 300


# ---- identity / versioning ---------------------------------------------------------------
def test_create_has_stable_ids_and_version_1(conn):
    r = add(conn, "b1", "Naming", "Use the product name Trueline exactly as written.")
    assert r.source_id.startswith("src_") and r.version_id.startswith("ver_")
    assert r.version == 1 and r.status == "active" and r.approval_status == "approved"


def test_update_creates_new_immutable_version_old_text_unchanged(conn, svc):
    r1 = add(conn, "b1", "Price", "The serum costs INR 1499.", stype="approved_fact")
    r2 = ingest.update_source(conn, EMB, SourceUpdate(brand_id="b1", source_id=r1.source_id,
                                                       text="The serum costs INR 1599."), S)
    assert r2.version == 2 and r2.version_id != r1.version_id
    old = ingest.get_version(conn, "b1", r1.source_id, r1.version_id)   # citation still resolves
    assert old.text == "The serum costs INR 1499." and old.status == "inactive"
    res = q(svc, "b1", "serum costs price")
    assert [x.version_id for x in res.results] == [r2.version_id]        # old version not retrievable
    assert "1599" in res.results[0].chunk_text


def test_quote_must_be_exact_substring_of_that_version(conn):
    r = add(conn, "b1", "Price", "The serum costs INR 1499.", stype="approved_fact")
    assert ingest.quote_exists(conn, "b1", r.source_id, r.version_id, "costs INR 1499")
    assert not ingest.quote_exists(conn, "b1", r.source_id, r.version_id, "costs INR 1599")


# ---- scoping / approval ----------------------------------------------------------------
def test_brand_isolation(conn, svc):
    add(conn, "b1", "Naming", "Always write Lumina Glow Serum.")
    add(conn, "b2", "Naming", "Always write Zenith Pro Max. Never use Lumina.")
    for brand, forbidden in (("b1", "Zenith"), ("b2", "Glow")):
        res = q(svc, brand, "product name Lumina")
        assert res.results and all(r.brand_id == brand for r in res.results)
        assert all(forbidden not in r.chunk_text for r in res.results)


def test_cross_brand_citation_lookup_hidden(conn):
    r = add(conn, "b1", "Naming", "Always write Lumina.")
    with pytest.raises(NotFound):
        ingest.get_version(conn, "b2", r.source_id, r.version_id)


def test_unapproved_and_deactivated_excluded(conn, svc):
    pend = add(conn, "b1", "Draft rule", "Zebra rule pending approval.", approved=False)
    assert pend.status == "pending" and pend.approval_status == "pending"
    assert not q(svc, "b1", "zebra rule").results
    ok = add(conn, "b1", "Live rule", "Giraffe rule that is live.")
    assert q(svc, "b1", "giraffe").results
    ingest.deactivate_source(conn, "b1", ok.source_id)
    assert not q(svc, "b1", "giraffe").results
    # historical version still resolvable
    assert ingest.get_version(conn, "b1", ok.source_id, ok.version_id).text.startswith("Giraffe")
    # admin approval makes the pending one searchable
    ingest.approve_version(conn, EMB, "b1", pend.version_id, S)
    assert q(svc, "b1", "zebra").results


def test_source_type_filter(conn, svc):
    add(conn, "b1", "Rule", "Pineapple wording rule.", stype="guideline")
    add(conn, "b1", "Fact", "Pineapple fact sheet entry.", stype="approved_fact")
    res = q(svc, "b1", "pineapple", source_types=["approved_fact"])
    assert {r.source_type for r in res.results} == {"approved_fact"}


# ---- result shape -------------------------------------------------------------------------
def test_result_has_citation_metadata_and_method(conn, svc):
    add(conn, "b1", "Naming", "Use the product name Trueline exactly as written.",
        metadata={"section": "Product naming", "page": 4}, document_name="Brand guide.pdf")
    res = q(svc, "b1", "which product name is allowed")
    assert res.retrieval_method == "hybrid"
    r = res.results[0]
    assert r.source_id and r.version_id and r.brand_id == "b1" and r.chunk_id
    assert r.chunk_text in r.source_text
    assert r.location["page"] == 4 and r.location["document_name"] == "Brand guide.pdf"
    assert r.source_text[r.location["char_start"]:r.location["char_end"]] == r.chunk_text
    assert r.rank == 1 and r.score is not None
    assert r.embedding_model == EMB.tag


def test_no_match_is_empty_success(conn, svc):
    add(conn, "b1", "Naming", "Use Trueline.")
    res = q(svc, "unknown_brand", "anything at all")
    assert not res.results and res.brand_id == "unknown_brand"


def test_top_k_bounds():
    with pytest.raises(ValidationError):
        KnowledgeSearchRequest(brand_id="b", query="x", top_k=0)
    with pytest.raises(ValidationError):
        KnowledgeSearchRequest(brand_id="b", query="x", top_k=21)
    with pytest.raises(ValidationError):
        KnowledgeSearchRequest(brand_id="", query="x")
    with pytest.raises(ValidationError):
        KnowledgeSearchRequest(brand_id="b", query="   ")
    assert KnowledgeSearchRequest(brand_id="b", query="x").top_k == 8


def test_top_k_limits_results(conn, svc):
    for i in range(6):
        add(conn, "b1", f"R{i}", f"Banana wording rule number {i}.")
    assert len(q(svc, "b1", "banana", top_k=3).results) == 3


# ---- failure behaviour ----------------------------------------------------------------------
class BrokenEmbedder(FakeEmbedder):
    tag = "broken"

    def embed(self, texts):
        raise RuntimeError("model unavailable")


def test_embedding_failure_is_visible_and_not_searchable(conn, svc):
    with pytest.raises(IndexingError):
        ingest.create_source(conn, BrokenEmbedder(), SourceCreate(
            brand_id="b1", source_type="guideline", title="X", text="Kiwi rule."), S)
    row = conn.execute("SELECT index_status FROM knowledge_chunks").fetchone()
    assert row["index_status"] == "failed"
    assert not q(svc, "b1", "kiwi").results
    # recovery: reindex with a working embedder
    ingest.reindex_stale(conn, EMB, S)
    assert q(svc, "b1", "kiwi").results


def test_query_embedding_failure_is_error_not_empty(conn):
    add(conn, "b1", "Naming", "Use Trueline.")
    bad = KnowledgeService(BrokenEmbedder(), S, conn_factory=lambda: psycopg.connect(URL, row_factory=dict_row))
    with pytest.raises(SearchServiceError):
        bad.search(KnowledgeSearchRequest(brand_id="b1", query="Trueline"))


def test_db_failure_is_error_not_empty(conn):
    bad = KnowledgeService(EMB, S, conn_factory=lambda: psycopg.connect(
        "postgresql://nobody:x@localhost:1/none", connect_timeout=2))
    with pytest.raises(SearchServiceError):
        bad.search(KnowledgeSearchRequest(brand_id="b1", query="x"))


def test_keyword_fallback_is_labelled_keyword(conn):
    from dataclasses import replace
    add(conn, "b1", "Naming", "Use Trueline exactly.")
    fb = KnowledgeService(BrokenEmbedder(), replace(S, allow_keyword_fallback=True),
                          conn_factory=lambda: psycopg.connect(URL, row_factory=dict_row))
    # fallback only searches chunks whose stored tag matches the (broken) embedder tag -> none;
    # point it at the tag the data was built with to prove the label:
    BrokenEmbedder.tag = EMB.tag
    try:
        res = fb.search(KnowledgeSearchRequest(brand_id="b1", query="Trueline"))
    finally:
        BrokenEmbedder.tag = "broken"
    assert res.retrieval_method == "keyword" and res.results
    assert all(r.score is not None for r in res.results)


def test_mixed_model_chunks_not_searched(conn, svc):
    add(conn, "b1", "Naming", "Mango rule.")
    conn.execute("UPDATE knowledge_chunks SET embedding_model='other-model@x|dense|normalized'")
    conn.commit()
    assert not q(svc, "b1", "mango").results
    assert ingest.reindex_stale(conn, EMB, S)           # rebuilds with current model tag
    assert q(svc, "b1", "mango").results
