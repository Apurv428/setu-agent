"""
retrieval.py — lightweight local RAG over the knowledge/ directory.

Uses paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers) so queries
in any Indian language can retrieve English passages. The index is built lazily
on first call to search() and cached to knowledge/.index.npz; it is rebuilt if
any document has been modified since the cache was written.

Public API:
    search(query: str, k: int = 3) -> list[dict]
    # each dict: {"source": str, "text": str, "score": float}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_CACHE_FILE = _KNOWLEDGE_DIR / ".index.npz"
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_chunks: list[tuple[str, str]] = []   # (source_filename, chunk_text)
_embeddings: np.ndarray | None = None
_model = None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_doc(path: Path) -> list[tuple[str, str]]:
    """Split a markdown doc into paragraphs and return (source, text) tuples."""
    text = path.read_text(encoding="utf-8")
    source = path.name
    results: list[tuple[str, str]] = []
    current: list[str] = []

    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        current.append(para)
        word_count = sum(len(p.split()) for p in current)
        if word_count >= 120:
            results.append((source, "\n\n".join(current)))
            current = []

    if current:
        chunk = "\n\n".join(current)
        if len(chunk) > 40:
            results.append((source, chunk))

    return results


def _collect_chunks() -> list[tuple[str, str]]:
    """Read all .md files and return (source, text) chunks."""
    docs = sorted(_KNOWLEDGE_DIR.glob("*.md")) + sorted(_KNOWLEDGE_DIR.glob("*.txt"))
    docs = [d for d in docs if not d.name.startswith(".")]
    if not docs:
        raise RuntimeError(
            f"No documents found in {_KNOWLEDGE_DIR}. "
            "Add .md or .txt files to the knowledge/ directory."
        )
    all_chunks: list[tuple[str, str]] = []
    for doc in docs:
        all_chunks.extend(_chunk_doc(doc))
    return all_chunks


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _max_doc_mtime() -> float:
    docs = list(_KNOWLEDGE_DIR.glob("*.md")) + list(_KNOWLEDGE_DIR.glob("*.txt"))
    docs = [d for d in docs if not d.name.startswith(".")]
    if not docs:
        return 0.0
    return max(d.stat().st_mtime for d in docs)


def _cache_is_fresh() -> bool:
    if not _CACHE_FILE.exists():
        return False
    cache_mtime = _CACHE_FILE.stat().st_mtime
    return _max_doc_mtime() <= cache_mtime


def _save_cache(chunks: list[tuple[str, str]], embeddings: np.ndarray) -> None:
    sources = np.array([c[0] for c in chunks])
    texts = np.array([c[1] for c in chunks])
    np.savez(_CACHE_FILE, embeddings=embeddings, sources=sources, texts=texts)


def _load_cache() -> tuple[list[tuple[str, str]], np.ndarray]:
    data = np.load(_CACHE_FILE, allow_pickle=True)
    sources = data["sources"].tolist()
    texts = data["texts"].tolist()
    embeddings = data["embeddings"]
    return list(zip(sources, texts)), embeddings


# ---------------------------------------------------------------------------
# Index bootstrap
# ---------------------------------------------------------------------------

def _ensure_index() -> None:
    global _chunks, _embeddings, _model

    if _model is not None:
        return

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from exc

    if _cache_is_fresh():
        print("  [retrieval] Loading index from cache...")
        _chunks, _embeddings = _load_cache()
    else:
        print("  [retrieval] Building index (this takes ~10s on first run)...")
        _chunks = _collect_chunks()
        if not _chunks:
            raise RuntimeError(
                f"knowledge/ has documents but no usable chunks were extracted. "
                "Ensure files contain paragraphs separated by blank lines."
            )
        _model = SentenceTransformer(_MODEL_NAME)
        texts = [c[1] for c in _chunks]
        _embeddings = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        _save_cache(_chunks, _embeddings)
        print(f"  [retrieval] Index ready: {len(_chunks)} chunks, cached to {_CACHE_FILE}")
        return  # _model already set above

    # Cache path: load model after loading cache
    _model = SentenceTransformer(_MODEL_NAME)
    print(f"  [retrieval] Index ready: {len(_chunks)} chunks")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(query: str, k: int = 3) -> list[dict]:
    """
    Return the top-k most relevant chunks for the query.

    Each result dict: {"source": str, "text": str, "score": float}
    Scores are cosine similarities in [0, 1].
    """
    _ensure_index()
    q_emb = _model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = _embeddings @ q_emb
    top_k = min(k, len(_chunks))
    indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"source": _chunks[i][0], "text": _chunks[i][1], "score": float(scores[i])}
        for i in indices
    ]


# ---------------------------------------------------------------------------
# CLI — python retrieval.py "your query"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retrieval.py \"your query\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    print(f"Query: {query}\n")
    results = search(query, k=3)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['source']} | score {r['score']:.3f}")
        print(r["text"])
        print()
