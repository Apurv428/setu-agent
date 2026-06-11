"""
tests/test_retrieval.py — unit tests for retrieval.py.

No network calls, no real model downloads. The embedding model is mocked
with a tiny synthetic matrix.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import retrieval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(embeddings: np.ndarray):
    """Return a SentenceTransformer mock whose encode() returns rows of embeddings."""
    model = MagicMock()
    call_count = [0]

    def fake_encode(texts, normalize_embeddings=True, show_progress_bar=False):
        # If encoding a single query (list of 1), return first row
        if isinstance(texts, list) and len(texts) == 1:
            return embeddings[:1]
        # Otherwise return as many rows as there are texts
        n = len(texts)
        return embeddings[:n]

    model.encode.side_effect = fake_encode
    return model


def _reset_retrieval_state():
    """Clear the module-level cache so each test starts fresh."""
    retrieval._chunks = []
    retrieval._embeddings = None
    retrieval._model = None


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

def test_chunking_produces_nonempty_chunks_with_sources(tmp_path):
    doc = tmp_path / "test_lang.md"
    doc.write_text(
        "# Test Language\n\n"
        "This is a paragraph about test language. " * 20 + "\n\n"
        "This is a second paragraph about scripts. " * 20,
        encoding="utf-8",
    )
    chunks = retrieval._chunk_doc(doc)
    assert len(chunks) >= 1
    for source, text in chunks:
        assert source == "test_lang.md"
        assert len(text) > 0


def test_chunking_skips_empty_paragraphs(tmp_path):
    doc = tmp_path / "sparse.md"
    doc.write_text("\n\n\n\nHello world\n\n\n\n", encoding="utf-8")
    chunks = retrieval._chunk_doc(doc)
    # "Hello world" is too short to form a real chunk on its own but should not error
    # (may be merged or kept; what matters is no crash and no empty text)
    for _source, text in chunks:
        assert text.strip() != ""


def test_collect_chunks_raises_on_empty_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval, "_KNOWLEDGE_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="No documents found"):
        retrieval._collect_chunks()


# ---------------------------------------------------------------------------
# Cosine similarity ordering
# ---------------------------------------------------------------------------

def test_cosine_top_k_ordering():
    """Verify search() returns results sorted by descending cosine similarity."""
    # 3 synthetic chunks; query should match chunk at index 1 most closely
    chunk_texts = ["chunk A", "chunk B", "chunk C"]
    chunk_sources = ["a.md", "b.md", "c.md"]
    chunks = list(zip(chunk_sources, chunk_texts))

    # Embeddings: each row is a unit vector; query vector aligns with row 1
    embs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    mock_model = MagicMock()
    call_count = [0]

    def fake_encode(texts, normalize_embeddings=True, show_progress_bar=False):
        call_count[0] += 1
        if len(texts) == 1:
            # Query: align with chunk B (index 1)
            return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        # Corpus
        return embs[:len(texts)]

    mock_model.encode.side_effect = fake_encode

    _reset_retrieval_state()
    retrieval._chunks = chunks
    retrieval._embeddings = embs
    retrieval._model = mock_model

    results = retrieval.search("query matching B", k=3)
    assert results[0]["source"] == "b.md"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert results[1]["score"] <= results[0]["score"]
    assert results[2]["score"] <= results[1]["score"]


def test_search_returns_dicts_with_required_keys():
    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

    _reset_retrieval_state()
    retrieval._chunks = [("src1.md", "text one"), ("src2.md", "text two")]
    retrieval._embeddings = embs
    retrieval._model = mock_model

    results = retrieval.search("anything", k=2)
    assert len(results) == 2
    for r in results:
        assert "source" in r
        assert "text" in r
        assert "score" in r
        assert isinstance(r["score"], float)


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

def test_cache_invalidation_on_newer_doc(tmp_path, monkeypatch):
    """_cache_is_fresh() returns False when a doc is newer than the cache."""
    monkeypatch.setattr(retrieval, "_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(retrieval, "_CACHE_FILE", tmp_path / ".index.npz")

    # Create a cache file
    cache = tmp_path / ".index.npz"
    cache.write_bytes(b"placeholder")

    # Create a doc that is clearly newer
    doc = tmp_path / "new_doc.md"
    doc.write_text("Some content here.\n", encoding="utf-8")

    # Set cache mtime to a past time
    old_time = time.time() - 100
    import os
    os.utime(cache, (old_time, old_time))

    # Doc mtime is now > cache mtime → not fresh
    assert retrieval._cache_is_fresh() is False


def test_cache_is_fresh_when_no_docs_newer(tmp_path, monkeypatch):
    """_cache_is_fresh() returns True when the cache is newer than all docs."""
    monkeypatch.setattr(retrieval, "_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(retrieval, "_CACHE_FILE", tmp_path / ".index.npz")

    doc = tmp_path / "old_doc.md"
    doc.write_text("Content.\n", encoding="utf-8")

    cache = tmp_path / ".index.npz"
    cache.write_bytes(b"placeholder")

    # Set doc to old time, cache to now
    old_time = time.time() - 100
    import os
    os.utime(doc, (old_time, old_time))

    assert retrieval._cache_is_fresh() is True


def test_cache_is_not_fresh_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval, "_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(retrieval, "_CACHE_FILE", tmp_path / ".index.npz")
    # No cache file created
    assert retrieval._cache_is_fresh() is False
