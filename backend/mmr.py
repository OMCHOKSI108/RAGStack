"""
Max Marginal Relevance (MMR) for diverse retrieval results.
Balances relevance to query with diversity among results
to avoid redundant information in the context.
"""

import logging
import numpy as np
from typing import List, Tuple

from backend.models import SearchResult

logger = logging.getLogger(__name__)


def mmr_select(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    chunks: list,
    top_k: int,
    lambda_mult: float = 0.5,
) -> List[Tuple[int, float]]:
    """
    Select top-k results using Max Marginal Relevance.
    
    MMR = argmax[ λ * Sim(query, doc_i) - (1-λ) * max(Sim(doc_i, doc_j)) ]
    
    Args:
        query_embedding: Normalized query embedding vector
        doc_embeddings: Matrix of normalized document embeddings (n x d)
        chunks: List of chunks corresponding to doc_embeddings
        top_k: Number of results to select
        lambda_mult: Trade-off between relevance (1.0) and diversity (0.0)
    
    Returns:
        List of (chunk_index, relevance_score) tuples in MMR order
    """
    if len(chunks) == 0:
        return []
    
    if len(chunks) <= top_k:
        # Not enough candidates to diversify, return all
        similarities = doc_embeddings @ query_embedding
        return [(i, float(similarities[i])) for i in range(len(chunks))]
    
    # Compute query-document similarities
    query_similarities = doc_embeddings @ query_embedding
    
    # Compute document-document similarity matrix
    doc_similarities = doc_embeddings @ doc_embeddings.T
    
    selected = []
    remaining = list(range(len(chunks)))
    
    # First: select the most relevant document
    best_idx = remaining[np.argmax([query_similarities[i] for i in remaining])]
    selected.append(best_idx)
    remaining.remove(best_idx)
    
    # Iteratively select remaining documents
    while len(selected) < top_k and remaining:
        best_mmr_score = -float('inf')
        best_candidate = None
        
        for candidate_idx in remaining:
            # Relevance to query
            relevance = query_similarities[candidate_idx]
            
            # Max similarity to already selected documents
            max_sim_to_selected = max(
                doc_similarities[candidate_idx, sel_idx]
                for sel_idx in selected
            )
            
            # MMR score
            mmr_score = (
                lambda_mult * relevance
                - (1 - lambda_mult) * max_sim_to_selected
            )
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate = candidate_idx
        
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
    
    # Return selected indices with their relevance scores
    return [(idx, float(query_similarities[idx])) for idx in selected]


def apply_mmr_to_results(
    search_results: List[SearchResult],
    query_embedding: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.5,
) -> List[SearchResult]:
    """
    Apply MMR to a list of SearchResult objects.
    
    Args:
        search_results: Initial search results (from hybrid search)
        query_embedding: Normalized query embedding
        top_k: Final number of results after MMR
        lambda_mult: Relevance-diversity trade-off
    
    Returns:
        MMR-diversified SearchResult list
    """
    if len(search_results) <= top_k:
        return search_results
    
    # Extract embeddings from chunks (need to re-encode)
    from backend import embeddings
    
    chunk_texts = [r.chunk.text for r in search_results]
    doc_embeddings = embeddings.encode(chunk_texts, normalize=True)
    
    # Apply MMR
    mmr_indices = mmr_select(
        query_embedding,
        doc_embeddings,
        chunk_texts,
        top_k,
        lambda_mult,
    )
    
    # Reorder results based on MMR selection
    mmr_results = []
    for new_rank, (orig_idx, relevance_score) in enumerate(mmr_indices):
        result = search_results[orig_idx]
        # Update score to reflect MMR ranking
        result.score = relevance_score
        result.global_index = new_rank
        mmr_results.append(result)
    
    logger.info(
        f"MMR applied: {len(search_results)} candidates -> "
        f"{len(mmr_results)} diverse results (λ={lambda_mult})"
    )
    
    return mmr_results
