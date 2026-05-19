"""
BM25 sparse search index.
Maintains a tokenized corpus aligned with the FAISS vector store
for hybrid retrieval via Reciprocal Rank Fusion.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from rank_bm25 import BM25Okapi

from backend.config import INDEX_DIR
from backend.models import DocumentChunk

logger = logging.getLogger(__name__)

BM25_FILE = INDEX_DIR / "bm25.pkl"


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class BM25Store:
    """
    BM25-based sparse search index.
    Keeps a parallel corpus of tokenized chunks with metadata
    for keyword-based retrieval.
    """

    def __init__(self):
        self.corpus: List[List[str]] = []       # Tokenized texts
        self.chunks: List[DocumentChunk] = []   # Corresponding chunk metadata
        self.doc_index: Dict[str, List[int]] = {}  # doc_id -> corpus positions
        self.bm25: Optional[BM25Okapi] = None

    @property
    def total_chunks(self) -> int:
        return len(self.corpus)

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from the current corpus."""
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        else:
            self.bm25 = None

    def add(self, chunks: List[DocumentChunk], doc_id: str) -> None:
        """Add tokenized chunks to the BM25 corpus."""
        positions = []
        start_idx = len(self.corpus)

        for i, chunk in enumerate(chunks):
            tokens = _tokenize(chunk.text)
            self.corpus.append(tokens)
            self.chunks.append(chunk)
            positions.append(start_idx + i)

        self.doc_index[doc_id] = positions
        self._rebuild_bm25()

        logger.info(
            f"BM25: Added {len(chunks)} chunks for doc_id={doc_id[:12]}... "
            f"(total: {len(self.corpus)})"
        )

    def remove_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to a document and rebuild."""
        if doc_id not in self.doc_index:
            return False

        positions_to_remove = set(self.doc_index[doc_id])

        new_corpus = []
        new_chunks = []
        new_doc_index: Dict[str, List[int]] = {}

        for old_pos in range(len(self.corpus)):
            if old_pos in positions_to_remove:
                continue

            new_pos = len(new_corpus)
            new_corpus.append(self.corpus[old_pos])
            new_chunks.append(self.chunks[old_pos])

            chunk_doc_id = self.chunks[old_pos].doc_id
            if chunk_doc_id not in new_doc_index:
                new_doc_index[chunk_doc_id] = []
            new_doc_index[chunk_doc_id].append(new_pos)

        self.corpus = new_corpus
        self.chunks = new_chunks
        self.doc_index = new_doc_index
        self._rebuild_bm25()

        logger.info(f"BM25: Removed doc_id={doc_id[:12]}..., {len(self.corpus)} chunks remaining")
        return True

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float, DocumentChunk]]:
        """
        Search for the top-k most relevant chunks using BM25 scoring.
        Returns list of (corpus_index, score, chunk) tuples.
        """
        if self.bm25 is None or not self.corpus:
            return []

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get top-k indices
        k = min(top_k, len(scores))
        top_indices = scores.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx]), self.chunks[idx]))

        return results

    def save(self) -> None:
        """Persist BM25 data to disk."""
        data = {
            "corpus": self.corpus,
            "chunks": [c.model_dump() for c in self.chunks],
            "doc_index": self.doc_index,
        }
        with open(BM25_FILE, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25: Saved {len(self.corpus)} chunks to disk")

    def load(self) -> bool:
        """Load BM25 data from disk. Returns True if loaded."""
        if not BM25_FILE.exists():
            logger.info("BM25: No existing index found")
            return False

        try:
            with open(BM25_FILE, "rb") as f:
                data = pickle.load(f)

            self.corpus = data["corpus"]
            self.chunks = [DocumentChunk(**c) for c in data["chunks"]]
            self.doc_index = data["doc_index"]
            self._rebuild_bm25()

            logger.info(f"BM25: Loaded {len(self.corpus)} chunks from disk")
            return True
        except Exception as e:
            logger.error(f"BM25: Failed to load index: {e}")
            return False
