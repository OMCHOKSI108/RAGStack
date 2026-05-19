"""
Query rewriter using the local LLM.
Makes minimal, safe improvements to queries for better retrieval.
"""

import logging

from backend import llm

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = (
    "You are a query optimizer for document search. "
    "Your ONLY job is to fix obvious typos. "
    "NEVER add words, NEVER expand acronyms, NEVER change meaning. "
    "Return ONLY the corrected query or the original if no typos."
)

REWRITE_USER = """Fix ONLY obvious typos in this query. Rules:
1. Fix misspelled words ONLY (e.g., "Offerleter" -> "Offer letter")
2. Do NOT add ANY words
3. Do NOT expand acronyms
4. Do NOT change meaning or intent
5. If no typos, return EXACTLY the original query
6. Return ONLY the query, nothing else

Query: {query}

Result:"""


def rewrite_query(query: str) -> str:
    """
    Rewrite a user query to improve retrieval quality.
    Falls back to the original query on any failure.
    """
    # Skip rewriting for very short queries
    if len(query.split()) <= 4:
        logger.info(f"Query too short, skipping rewrite: '{query}'")
        return query

    # Skip if query already looks well-formed
    if query.endswith('?') and len(query.split()) <= 8:
        logger.info(f"Simple question, skipping rewrite: '{query}'")
        return query

    try:
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": REWRITE_USER.format(query=query)}
        ]

        rewritten = llm.generate(
            messages=messages,
            max_tokens=30,
            temperature=0.0,
        )

        # Basic validation: rewritten query should not be empty or too long
        rewritten = rewritten.strip().strip('"').strip("'")
        if not rewritten or len(rewritten) > 150:
            logger.warning("Query rewriting produced invalid output, using original")
            return query

        # If rewritten is drastically different, reject it
        original_words = set(query.lower().split())
        rewritten_words = set(rewritten.lower().split())
        overlap = len(original_words & rewritten_words)
        
        if overlap < len(original_words) * 0.7:
            logger.warning(f"Rewritten query too different from original, using original: '{rewritten}'")
            return query

        if rewritten.lower() == query.lower():
            logger.info("Query unchanged after rewriting")
            return query

        logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}, using original query")
        return query
