"""
Embedding generation, isolated behind a single function so the model can be
swapped (e.g. to a hosted embeddings API) without touching calling code.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Loaded once per process; multilingual so Hindi + English text both embed well.
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return vectors.tolist()


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 150) -> list[str]:
    """Simple overlapping chunker for legal text. Good enough for MVP; swap for a
    sentence/paragraph-aware splitter later if legal clause boundaries matter more."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return [c.strip() for c in chunks if c.strip()]
