"""
Embedding wrapper with local SentenceTransformer and Hugging Face API providers.
"""

from __future__ import annotations

import logging

import numpy as np

from backend.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER

logger = logging.getLogger(__name__)

_model = None


def get_model():
    """Load or return the cached local embedding model."""
    global _model
    if EMBEDDING_PROVIDER == "HUGGINGFACE_API":
        return None

    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model: %s on %s", EMBEDDING_MODEL, device)
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        logger.info("Embedding model loaded")
    return _model


def encode(texts: list[str], normalize: bool = True) -> np.ndarray:
    """Encode texts into dense vectors."""
    if EMBEDDING_PROVIDER == "HUGGINGFACE_API":
        from backend.hf_api import embed_texts

        return embed_texts(texts, normalize=normalize)

    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

    return embeddings


def encode_query(query: str, normalize: bool = True) -> np.ndarray:
    """Encode a single query string."""
    return encode([query], normalize=normalize)
