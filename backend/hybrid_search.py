"""
Hybrid search combining dense retrieval, optional BM25 sparse retrieval,
RRF fusion, MMR, and local or remote re-ranking.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from backend import embeddings
from backend.config import (
    RERANK_PROVIDER,
    RERANK_TOP_K,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
    RRF_K,
    SPARSE_RETRIEVAL,
)
from backend.mmr import apply_mmr_to_results
from backend.models import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    """Load or return the cached local cross-encoder reranker."""
    global _reranker
    if RERANK_PROVIDER == "HUGGINGFACE_API":
        return None

    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading re-ranker model: %s on %s", RERANKER_MODEL, device)
        _reranker = CrossEncoder(RERANKER_MODEL, device=device)
        logger.info("Re-ranker loaded")
    return _reranker


def reciprocal_rank_fusion(ranked_lists: List[List[tuple]], k: int = RRF_K) -> List[tuple]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    fused_scores: Dict[str, float] = {}
    fused_data: Dict[str, DocumentChunk] = {}

    for ranked_list in ranked_lists:
        for rank, (identifier, chunk) in enumerate(ranked_list):
            if identifier not in fused_scores:
                fused_scores[identifier] = 0.0
                fused_data[identifier] = chunk
            fused_scores[identifier] += 1.0 / (k + rank + 1)

    sorted_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [(ident, score, fused_data[ident]) for ident, score in sorted_results]


def _rerank_candidates(query: str, candidates: List[tuple]) -> List[float]:
    """Return rerank scores, falling back to existing retrieval scores on errors."""
    if not candidates:
        return []

    try:
        if RERANK_PROVIDER == "HUGGINGFACE_API":
            from backend.hf_api import rerank

            return rerank(query, [chunk.text for _, _, chunk in candidates])

        reranker = get_reranker()
        pairs = [(query, chunk.text) for _, _, chunk in candidates]
        return list(reranker.predict(pairs))
    except Exception as exc:
        logger.warning("Reranker failed; using retrieval scores: %s", exc)
        return [score for _, score, _ in candidates]


def hybrid_search(
    query: str,
    vector_store,
    bm25_store,
    top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    use_mmr: bool = False,
    mmr_lambda: float = 0.5,
    mmr_top_k: int = None,
) -> List[SearchResult]:
    """Perform dense search, optional sparse fusion, optional MMR, and reranking."""
    query_embedding = embeddings.encode_query(query)
    dense_results = vector_store.search(query_embedding, top_k=top_k)
    dense_ranked = [
        (f"{result.chunk.doc_id}_{result.chunk.chunk_index}", result.chunk)
        for result in dense_results
    ]

    ranked_lists = [dense_ranked]
    if SPARSE_RETRIEVAL and bm25_store is not None:
        sparse_results = bm25_store.search(query, top_k=top_k)
        ranked_lists.append([
            (f"{chunk.doc_id}_{chunk.chunk_index}", chunk)
            for _, _, chunk in sparse_results
        ])

    fused = reciprocal_rank_fusion(ranked_lists)
    logger.info("RRF fusion: %s unique candidates", len(fused))
    if not fused:
        return []

    if use_mmr and len(fused) > rerank_top_k:
        fused_results = [
            SearchResult(chunk=chunk, score=float(score), global_index=i)
            for i, (_, score, chunk) in enumerate(fused)
        ]
        fused_results = apply_mmr_to_results(
            fused_results,
            query_embedding.flatten(),
            mmr_top_k or min(top_k, len(fused)),
            mmr_lambda,
        )
        fused = [
            (f"{result.chunk.doc_id}_{result.chunk.chunk_index}", result.score, result.chunk)
            for result in fused_results
        ]

    candidates = fused[:top_k]
    rerank_scores = _rerank_candidates(query, candidates)
    reranked = [
        SearchResult(chunk=chunk, score=float(rerank_scores[i]), global_index=i)
        for i, (_, _, chunk) in enumerate(candidates)
    ]
    reranked.sort(key=lambda result: result.score, reverse=True)
    return reranked[:rerank_top_k]


def multi_query_hybrid_search(
    queries: List[str],
    vector_store,
    bm25_store,
    top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
) -> List[SearchResult]:
    """Run hybrid search for multiple queries and deduplicate final candidates."""
    all_results = []
    for i, query in enumerate(queries):
        logger.info("Multi-query search [%s/%s]: %s", i + 1, len(queries), query)
        all_results.append(
            hybrid_search(
                query,
                vector_store,
                bm25_store,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
            )
        )

    seen_ids = set()
    fused = []
    for results in all_results:
        for result in results:
            chunk_id = f"{result.chunk.doc_id}_{result.chunk.chunk_index}"
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            fused.append(result)
            if len(fused) >= rerank_top_k * 2:
                break
        if len(fused) >= rerank_top_k * 2:
            break

    if len(fused) > rerank_top_k:
        primary_query = queries[0]
        candidates = [
            (f"{result.chunk.doc_id}_{result.chunk.chunk_index}", result.score, result.chunk)
            for result in fused[:top_k]
        ]
        rerank_scores = _rerank_candidates(primary_query, candidates)
        for i, result in enumerate(fused[:top_k]):
            result.score = float(rerank_scores[i])
        fused.sort(key=lambda result: result.score, reverse=True)
        fused = fused[:rerank_top_k]

    for i, result in enumerate(fused):
        result.global_index = i

    logger.info("Multi-query fusion produced %s unique results", len(fused))
    return fused
