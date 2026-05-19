"""
Context compression: reduces retrieved context to only the most relevant
passages before sending to the LLM. Uses sentence-level relevance scoring
to filter out noise while preserving key information.
"""

import logging
from typing import List, Dict

import numpy as np

from backend import embeddings

logger = logging.getLogger(__name__)


def compress_context(
    context_chunks: List[Dict[str, str]],
    query: str,
    compression_ratio: float = 0.7,
    min_chunks: int = 3,
) -> List[Dict[str, str]]:
    """
    Compress context by scoring each chunk's relevance to the query
    and keeping only the most relevant ones.
    
    Args:
        context_chunks: List of {"text", "source_file", "page_number"} dicts
        query: The user's question
        compression_ratio: Fraction of chunks to keep (0.0-1.0)
        min_chunks: Minimum number of chunks to always keep
    
    Returns:
        Filtered list of context chunks, ordered by relevance
    """
    if len(context_chunks) <= min_chunks:
        return context_chunks
    
    try:
        # Encode query and all chunks
        query_embedding = embeddings.encode([query], normalize=True)
        chunk_texts = [c["text"] for c in context_chunks]
        chunk_embeddings = embeddings.encode(chunk_texts, normalize=True)
        
        # Compute relevance scores
        similarities = (chunk_embeddings @ query_embedding.T).flatten()
        
        # Score each chunk
        scored_chunks = []
        for i, chunk in enumerate(context_chunks):
            scored_chunks.append({
                **chunk,
                "_relevance_score": float(similarities[i]),
                "_original_index": i,
            })
        
        # Sort by relevance descending
        scored_chunks.sort(key=lambda x: x["_relevance_score"], reverse=True)
        
        # Keep top chunks based on compression ratio
        keep_count = max(min_chunks, int(len(context_chunks) * compression_ratio))
        kept = scored_chunks[:keep_count]
        
        # Restore original order for coherence
        kept.sort(key=lambda x: x["_original_index"])
        
        # Remove internal scoring fields
        for chunk in kept:
            chunk.pop("_relevance_score", None)
            chunk.pop("_original_index", None)
        
        removed = len(context_chunks) - len(kept)
        logger.info(
            f"Context compression: {len(context_chunks)} chunks -> "
            f"{len(kept)} chunks (removed {removed}, ratio={compression_ratio})"
        )
        
        return kept
        
    except Exception as e:
        logger.warning(f"Context compression failed: {e}, returning all chunks")
        return context_chunks


def extract_key_sentences(
    text: str,
    query: str,
    max_sentences: int = 5,
) -> str:
    """
    Extract the most relevant sentences from a single text chunk.
    Useful for compressing individual large chunks.
    """
    # Split into sentences
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in '.!?' and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    
    if len(sentences) <= max_sentences:
        return text
    
    try:
        # Score each sentence
        query_embedding = embeddings.encode([query], normalize=True)
        sent_embeddings = embeddings.encode(sentences, normalize=True)
        scores = (sent_embeddings @ query_embedding.T).flatten()
        
        # Select top sentences
        top_indices = np.argsort(scores)[-max_sentences:][::-1]
        selected = [sentences[i] for i in sorted(top_indices)]
        
        return " ".join(selected)
        
    except Exception as e:
        logger.warning(f"Sentence extraction failed: {e}")
        return text
