"""
Query result caching layer.
Caches complete query responses to avoid redundant computation
for repeated or similar questions.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from backend.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "index" / "query_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 3600  # 1 hour default TTL
MAX_CACHE_SIZE = 100  # Maximum number of cached responses


class QueryCache:
    """Simple file-based cache for query responses."""
    
    def __init__(self, ttl: int = CACHE_TTL, max_size: int = MAX_CACHE_SIZE):
        self.ttl = ttl
        self.max_size = max_size
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, question: str, doc_count: int) -> str:
        """Generate a cache key from question and document state."""
        content = f"{question.lower().strip()}|docs:{doc_count}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.json"
    
    def get(self, question: str, doc_count: int) -> Optional[Dict]:
        """Retrieve cached response if available and not expired."""
        key = self._get_cache_key(question, doc_count)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            # Check TTL
            if time.time() - cached.get("timestamp", 0) > self.ttl:
                cache_path.unlink()
                logger.info(f"Cache expired for key: {key[:8]}...")
                return None
            
            logger.info(f"Cache hit for query: {question[:50]}...")
            return cached.get("response")
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def put(self, question: str, doc_count: int, response: Dict):
        """Store response in cache."""
        key = self._get_cache_key(question, doc_count)
        cache_path = self._get_cache_path(key)
        
        try:
            cached_data = {
                "question": question,
                "timestamp": time.time(),
                "doc_count": doc_count,
                "response": response,
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cached_data, f)
            
            logger.info(f"Cached response for key: {key[:8]}...")
            self._enforce_max_size()
            
        except IOError as e:
            logger.warning(f"Cache write error: {e}")
    
    def _enforce_max_size(self):
        """Remove oldest entries if cache exceeds max size."""
        cache_files = list(CACHE_DIR.glob("*.json"))
        if len(cache_files) > self.max_size:
            # Sort by modification time, oldest first
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            for old_file in cache_files[:len(cache_files) - self.max_size]:
                old_file.unlink()
            logger.info(f"Cache cleaned: removed {len(cache_files) - self.max_size} old entries")
    
    def clear(self):
        """Clear all cached responses."""
        for cache_file in CACHE_DIR.glob("*.json"):
            cache_file.unlink()
        logger.info("Query cache cleared")


# Singleton instance
_cache: Optional[QueryCache] = None


def get_cache() -> QueryCache:
    """Get or create the query cache singleton."""
    global _cache
    if _cache is None:
        _cache = QueryCache()
    return _cache
