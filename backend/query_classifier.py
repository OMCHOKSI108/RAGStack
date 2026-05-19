"""
Query classification: categorizes user queries to optimize retrieval strategy.
Different query types benefit from different retrieval approaches.
"""

import logging

from backend import llm

logger = logging.getLogger(__name__)

CLASSIFICATION_SYSTEM = (
    "You are a query classifier. Classify the user's question into ONE "
    "of these categories. Return ONLY the category name."
)

CLASSIFICATION_PROMPT = """Classify this query into exactly ONE category:

Categories:
- factual: Asks for specific facts, dates, names, definitions (e.g., "What is RAG?")
- analytical: Requires analysis, comparison, or synthesis (e.g., "Compare X and Y")
- procedural: Asks for steps, instructions, or how-to (e.g., "How to build a RAG system?")
- summarization: Asks for a summary or overview (e.g., "Summarize this document")
- conversational: General chat or follow-up (e.g., "Thanks", "Tell me more")

Query: {query}

Category:"""

VALID_CATEGORIES = {"factual", "analytical", "procedural", "summarization", "conversational"}

# Configuration per query type
QUERY_STRATEGIES = {
    "factual": {
        "use_multi_query": False,
        "mmr_lambda": 0.3,  # Prioritize diversity less for factual
        "compression_ratio": 0.8,  # Keep more context
        "top_k": 10,
        "rerank_top_k": 5,
    },
    "analytical": {
        "use_multi_query": True,
        "mmr_lambda": 0.5,  # Balance relevance and diversity
        "compression_ratio": 0.7,
        "top_k": 15,
        "rerank_top_k": 7,
    },
    "procedural": {
        "use_multi_query": True,
        "mmr_lambda": 0.4,
        "compression_ratio": 0.6,
        "top_k": 12,
        "rerank_top_k": 6,
    },
    "summarization": {
        "use_multi_query": False,
        "mmr_lambda": 0.6,  # More diversity for broad coverage
        "compression_ratio": 0.5,
        "top_k": 15,
        "rerank_top_k": 8,
    },
    "conversational": {
        "use_multi_query": False,
        "mmr_lambda": 0.5,
        "compression_ratio": 0.8,
        "top_k": 5,
        "rerank_top_k": 3,
    },
}


def classify_query(query: str) -> dict:
    """
    Classify a query and return the category with recommended strategy.

    Returns:
        {"category": str, "strategy": dict}
    """
    # Skip classification for very short queries
    if len(query.split()) <= 2:
        return {
            "category": "conversational",
            "strategy": QUERY_STRATEGIES["conversational"],
        }

    try:
        messages = [
            {"role": "system", "content": CLASSIFICATION_SYSTEM},
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(query=query)},
        ]

        category = llm.generate(
            messages=messages,
            max_tokens=20,
            temperature=0.0,
        ).strip().lower()

        # Validate category
        if category not in VALID_CATEGORIES:
            logger.warning(f"Invalid category '{category}', defaulting to factual")
            category = "factual"

        strategy = QUERY_STRATEGIES[category]
        logger.info(f"Query classified as '{category}'")

        return {"category": category, "strategy": strategy}

    except Exception as e:
        logger.warning(f"Query classification failed: {e}, using factual strategy")
        return {
            "category": "factual",
            "strategy": QUERY_STRATEGIES["factual"],
        }
