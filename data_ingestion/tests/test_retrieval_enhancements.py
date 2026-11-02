from trueline_ingestion.search import _parent_context, _rerank_candidates


def test_parent_context_returns_surrounding_paragraph():
    source = "Heading\n\nFirst rule in this paragraph. Second rule in this paragraph.\n\nOther section."
    child = "Second rule in this paragraph."
    start = source.index(child)
    context = _parent_context(source, {"char_start": start, "char_end": start + len(child)})

    assert child in context
    assert "First rule in this paragraph." in context
    assert "Other section." not in context


def test_parent_context_is_bounded_and_contains_child():
    source = "word " * 1500
    child = source[3000:3300]
    context = _parent_context(source, {"char_start": 3000, "char_end": 3300}, max_chars=500)

    assert len(context) <= 500
    assert child in context


def test_cross_encoder_reranking_preserves_rrf_scores(monkeypatch):
    class FakeScorer:
        def predict(self, pairs):
            return [0.2 if "wrong" in text else 0.9 for _, text in pairs]

    first = {"row": {"chunk_id": "a", "chunk_text": "wrong passage"}, "score": 0.03}
    second = {"row": {"chunk_id": "b", "chunk_text": "relevant passage"}, "score": 0.02}
    monkeypatch.setattr("trueline_ingestion.search._load_reranker", lambda _name: FakeScorer())

    ordered, scores = _rerank_candidates("query", [first, second], "test-reranker")

    assert [item["row"]["chunk_id"] for item in ordered] == ["b", "a"]
    assert scores == [0.9, 0.2]
    assert [item["score"] for item in ordered] == [0.02, 0.03]
