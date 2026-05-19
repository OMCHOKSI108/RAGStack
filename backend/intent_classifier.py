"""
Intent classifier for intelligent document reasoning.
Detects what the user wants to do with the document:
- extraction: pull specific data/entities from text
- counting: count occurrences of something
- comparison: compare two or more things
- summarization: summarize content
- verification: verify if something exists or is true
- qa: general question answering
- table_parsing: extract tabular data
- out_of_scope: math, personal, general knowledge (hard reject)
"""

import logging
import re
from typing import Dict, List

from backend import llm

logger = logging.getLogger(__name__)

# Out-of-scope detection patterns
_OUT_OF_SCOPE_PATTERNS = [
    # Math/calculation queries unrelated to documents
    (re.compile(r'\b(calculate|compute|solve|evaluate)\b.*\b(equation|integral|derivative|matrix|polynomial)\b', re.I), "math"),
    (re.compile(r'\bwhat\s+is\s+\d+\s*[\+\-\*/\^]\s*\d+', re.I), "math"),
    # Personal queries
    (re.compile(r'\b(my|your|his|her)\s+(name|age|birthday|address|phone|email)\b', re.I), "personal"),
    (re.compile(r'\bwho\s+am\s+i\b', re.I), "personal"),
    # General knowledge clearly unrelated to documents
    (re.compile(r'\b(capital|president|prime minister|population|gdp)\s+of\s+(france|china|india|usa|japan|germany|brazil)', re.I), "general"),
    (re.compile(r'\b(who\s+won|what\s+happened)\s+(in\s+)?(the\s+)?(world\s+cup|olympics|super\s+bowl|election)', re.I), "general"),
    (re.compile(r'\b(explain|what\s+is|define)\s+(quantum\s+physics|theory\s+of\s+relativity|general\s+relativity|special\s+relativity|black\s+hole)', re.I), "general"),
]

INTENT_SYSTEM = (
    "You are an intent classifier for document analysis. "
    "Classify the user's request into EXACTLY ONE category. "
    "Return ONLY the category name."
)

INTENT_PROMPT = """Classify this user request into exactly ONE category:

Categories:
- extraction: Pull specific data, entities, names, numbers, dates from document (e.g., "extract all companies", "get the MSME number", "list all names")
- counting: Count occurrences or items (e.g., "how many companies", "count the projects", "total number of")
- comparison: Compare two or more things (e.g., "compare X and Y", "what's the difference between", "which is better")
- summarization: Summarize or overview content (e.g., "summarize this", "what is this about", "give me an overview")
- verification: Verify if something exists or is true in document (e.g., "is X mentioned", "does the document say", "verify that")
- table_parsing: Extract tabular or structured data (e.g., "extract the table", "get the data in table format")
- qa: General question answering about the document (e.g., "tell me about X", "what is Y", "explain Z")
- out_of_scope: Math problems, personal questions, or general knowledge not related to uploaded documents

Rules:
- If the request is a math problem, personal question, or general world knowledge → out_of_scope
- If the request mentions "extract", "list all", "get all", "pull" → extraction
- If the request mentions "how many", "count", "total number" → counting
- If the request mentions "compare", "difference", "vs" → comparison
- If the request mentions "summarize", "overview", "what is this about" → summarization
- If the request mentions "is", "does", "verify", "check if" → verification
- If the request mentions "table", "tabular", "structured data" → table_parsing
- Otherwise → qa

Request: {query}

Category:"""

VALID_INTENTS = {
    "extraction",
    "counting",
    "comparison",
    "summarization",
    "verification",
    "table_parsing",
    "qa",
    "out_of_scope",
}

# Intent-specific retrieval and processing strategies
INTENT_STRATEGIES = {
    "extraction": {
        "top_k": 15,
        "rerank_top_k": 8,
        "use_multi_query": True,
        "compression_ratio": 0.6,
        "structured_output": True,
        "requires_verification": True,
    },
    "counting": {
        "top_k": 12,
        "rerank_top_k": 6,
        "use_multi_query": True,
        "compression_ratio": 0.5,
        "structured_output": True,
        "requires_verification": True,
    },
    "comparison": {
        "top_k": 15,
        "rerank_top_k": 8,
        "use_multi_query": True,
        "compression_ratio": 0.7,
        "structured_output": False,
        "requires_verification": True,
    },
    "summarization": {
        "top_k": 10,
        "rerank_top_k": 5,
        "use_multi_query": False,
        "compression_ratio": 0.8,
        "structured_output": False,
        "requires_verification": False,
    },
    "verification": {
        "top_k": 8,
        "rerank_top_k": 4,
        "use_multi_query": False,
        "compression_ratio": 0.9,
        "structured_output": True,
        "requires_verification": True,
    },
    "table_parsing": {
        "top_k": 10,
        "rerank_top_k": 5,
        "use_multi_query": False,
        "compression_ratio": 0.7,
        "structured_output": True,
        "requires_verification": True,
    },
    "qa": {
        "top_k": 12,
        "rerank_top_k": 8,
        "use_multi_query": False,
        "compression_ratio": 1.0,
        "structured_output": False,
        "requires_verification": False,
    },
    "out_of_scope": {
        "top_k": 0,
        "rerank_top_k": 0,
        "use_multi_query": False,
        "compression_ratio": 0.0,
        "structured_output": False,
        "requires_verification": False,
    },
}


def classify_intent(query: str) -> Dict:
    """
    Classify user intent and return recommended strategy.
    
    Returns:
        {"intent": str, "strategy": dict, "extracted_params": dict, "is_out_of_scope": bool}
    """
    query_lower = query.lower()
    
    # Hard rejection: check out-of-scope patterns FIRST
    for pattern, scope_type in _OUT_OF_SCOPE_PATTERNS:
        if pattern.search(query):
            logger.info(f"Out-of-scope detected ({scope_type}): '{query[:80]}'")
            return {
                "intent": "out_of_scope",
                "strategy": INTENT_STRATEGIES["out_of_scope"],
                "params": {"scope_type": scope_type},
                "is_out_of_scope": True,
            }
    
    # Fast heuristic classification for common patterns
    if any(kw in query_lower for kw in ["extract", "list all", "get all", "pull", "give me list", "give me all"]):
        intent = "extraction"
    elif any(kw in query_lower for kw in ["how many", "count", "total number", "number of"]):
        intent = "counting"
    elif any(kw in query_lower for kw in ["compare", "difference between", " vs ", "versus"]):
        intent = "comparison"
    elif any(kw in query_lower for kw in ["summarize", "overview", "what is this document", "what is this about"]):
        intent = "summarization"
    elif any(kw in query_lower for kw in ["is there", "does the document", "verify", "check if", "is mentioned"]):
        intent = "verification"
    elif any(kw in query_lower for kw in ["table", "tabular", "in table format"]):
        intent = "table_parsing"
    else:
        intent = "qa"
    
    # For complex queries, use LLM to confirm
    if len(query.split()) > 8:
        try:
            messages = [
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": INTENT_PROMPT.format(query=query)},
            ]
            
            llm_intent = llm.generate(
                messages=messages,
                max_tokens=20,
                temperature=0.0,
            ).strip().lower()
            
            if llm_intent in VALID_INTENTS:
                intent = llm_intent
                logger.info(f"LLM classified intent as '{intent}'")
            else:
                logger.info(f"LLM returned invalid intent '{llm_intent}', using heuristic '{intent}'")
                
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}, using heuristic '{intent}'")
    
    strategy = INTENT_STRATEGIES[intent]
    extracted_params = _extract_query_params(query_lower, intent)
    
    logger.info(f"Intent classified: '{intent}' (strategy: {strategy})")
    
    return {
        "intent": intent,
        "strategy": strategy,
        "params": extracted_params,
        "is_out_of_scope": intent == "out_of_scope",
    }


def _extract_query_params(query: str, intent: str) -> Dict:
    """Extract useful parameters from the query for task pipelines."""
    params = {}
    
    # Extract max limit (e.g., "maximum 40", "max 10", "up to 5")
    import re
    max_match = re.search(r'(?:maximum|max|up to|at most)\s+(\d+)', query)
    if max_match:
        params["max_limit"] = int(max_match.group(1))
    
    # Extract "all" flag
    if "all" in query or "every" in query:
        params["extract_all"] = True
    
    # Extract target entity for extraction/counting
    if intent in ("extraction", "counting"):
        # Simple heuristic: get the noun phrase after keywords
        for kw in ["extract", "list", "get", "pull", "count", "how many"]:
            if kw in query:
                idx = query.find(kw)
                after = query[idx + len(kw):].strip()
                # Remove common filler words
                for filler in ["the", "all", "from", "document", "pdf", "file"]:
                    after = after.replace(filler, "").strip()
                if after:
                    params["target_entity"] = after
                    break
    
    return params
