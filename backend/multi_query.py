"""
Multi-query retrieval: generates alternate versions of the user query
to improve recall by searching from different semantic angles.
Results are deduplicated and fused.
"""

import logging
from typing import List

from backend import llm

logger = logging.getLogger(__name__)

MULTI_QUERY_SYSTEM = (
    "You are a query expansion specialist. Generate 3 alternate versions "
    "of the user's question that preserve the original intent but use "
    "different wording, synonyms, or perspectives. "
    "Return ONLY the 3 queries, one per line, with no numbering or extra text."
)

MULTI_QUERY_PROMPT = """Generate 3 alternate search queries for this question.
Rules:
1. Preserve the original meaning and intent
2. Use different wording, synonyms, or perspectives
3. Each query should be self-contained
4. Return ONLY the 3 queries, one per line

Original: {query}

Alternates:"""


def generate_alternate_queries(
    query: str,
    num_alternates: int = 3,
) -> List[str]:
    """
    Generate alternate query versions for multi-query retrieval.
    
    Returns list including original query + alternates.
    """
    # Skip for very short queries
    if len(query.split()) <= 2:
        logger.info(f"Query too short for multi-query expansion: '{query}'")
        return [query]
    
    try:
        messages = [
            {"role": "system", "content": MULTI_QUERY_SYSTEM},
            {"role": "user", "content": MULTI_QUERY_PROMPT.format(query=query)},
        ]
        
        response = llm.generate(
            messages=messages,
            max_tokens=150,
            temperature=0.3,
        )
        
        # Parse the response into individual queries
        alternates = []
        for line in response.strip().split('\n'):
            line = line.strip().strip('"').strip("'").strip('-').strip()
            # Remove numbering like "1.", "2.", etc.
            if line and len(line) > 5:
                # Remove leading numbers/bullets
                cleaned = line.lstrip('0123456789.-) ')
                if cleaned and cleaned.lower() != query.lower():
                    alternates.append(cleaned)
        
        # Deduplicate and limit
        seen = set()
        unique_alternates = []
        for alt in alternates:
            if alt.lower() not in seen and len(unique_alternates) < num_alternates:
                seen.add(alt.lower())
                unique_alternates.append(alt)
        
        queries = [query] + unique_alternates
        logger.info(
            f"Multi-query expansion: '{query}' -> {len(queries)} queries"
        )
        return queries
        
    except Exception as e:
        logger.warning(f"Multi-query generation failed: {e}, using original")
        return [query]


def fuse_multi_query_results(
    all_results: List[List],
    max_results: int = 20,
) -> List:
    """
    Fuse results from multiple queries, deduplicating by chunk ID.
    Uses a simple scoring: first appearance gets higher weight.
    """
    seen_ids = set()
    fused = []
    
    for results in all_results:
        for result in results:
            chunk_id = f"{result.chunk.doc_id}_{result.chunk.chunk_index}"
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                fused.append(result)
                if len(fused) >= max_results:
                    break
        if len(fused) >= max_results:
            break
    
    logger.info(
        f"Multi-query fusion: {sum(len(r) for r in all_results)} total -> "
        f"{len(fused)} unique results"
    )
    
    return fused
