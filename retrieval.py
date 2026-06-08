"""
retrieval.py — lightweight local RAG over the knowledge/ directory.

Uses paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers) so queries
in any Indian language retrieve the right passages. The index is built once on
the first call to search() and kept in memory. No external vector database needed.
"""

from pathlib import Path
import numpy as np

_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_chunks: list[str] = []
_embeddings: np.ndarray | None = None
_model = None


def _load_chunks() -> list[str]:
    paths = sorted(_KNOWLEDGE_DIR.glob("*.md")) + sorted(_KNOWLEDGE_DIR.glob("*.txt"))
    chunks = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) > 50:
                chunks.append(para)
    return chunks


def _ensure_index() -> None:
    global _chunks, _embeddings, _model
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer
    print("  [retrieval] Loading embedding model (first call only)...")
    _model = SentenceTransformer(_MODEL_NAME)
    _chunks = _load_chunks()
    if not _chunks:
        raise RuntimeError(
            f"No documents found in {_KNOWLEDGE_DIR}. Add .md or .txt files."
        )
    _embeddings = _model.encode(_chunks, normalize_embeddings=True)
    print(f"  [retrieval] Index ready: {len(_chunks)} passages from {_KNOWLEDGE_DIR}")


def search(query: str, k: int = 3) -> list[str]:
    """Return the top-k most relevant passages for the query."""
    _ensure_index()
    q_emb = _model.encode([query], normalize_embeddings=True)[0]
    scores = _embeddings @ q_emb
    top_k = min(k, len(_chunks))
    indices = np.argsort(scores)[::-1][:top_k]
    return [_chunks[i] for i in indices]
