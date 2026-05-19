"""
FAISS vector store with document-level metadata management.
Supports add, remove, search, save, and load operations
with full document versioning via doc_id tracking.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np

from backend.config import EMBEDDING_DIMENSION, INDEX_DIR
from backend.models import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.pkl"


class VectorStore:
    """
    FAISS-backed vector store with document-level metadata.
    Uses IndexFlatIP (inner product) with normalized vectors for cosine similarity.
    """

    def __init__(self):
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        # Maps FAISS position -> DocumentChunk
        self.metadata: Dict[int, DocumentChunk] = {}
        # Maps doc_id (file hash) -> list of FAISS positions
        self.doc_index: Dict[str, List[int]] = {}
        # Maps doc_id -> filename for reverse lookup
        self.doc_filenames: Dict[str, str] = {}

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal

    @property
    def total_documents(self) -> int:
        return len(self.doc_index)

    def add(
        self,
        embeddings: np.ndarray,
        chunks: List[DocumentChunk],
        doc_id: str
    ) -> None:
        """
        Add document chunks and their embeddings to the store.
        Tracks which chunks belong to which document via doc_id.
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length")

        start_idx = self.index.ntotal
        self.index.add(embeddings)

        positions = []
        for i, chunk in enumerate(chunks):
            pos = start_idx + i
            self.metadata[pos] = chunk
            positions.append(pos)

        self.doc_index[doc_id] = positions
        if chunks:
            self.doc_filenames[doc_id] = chunks[0].source_file

        logger.info(
            f"Added {len(chunks)} chunks for doc_id={doc_id[:12]}... "
            f"(total: {self.index.ntotal} chunks)"
        )

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.
        Rebuilds the FAISS index without the removed chunks.
        """
        if doc_id not in self.doc_index:
            logger.warning(f"Document {doc_id[:12]}... not found in index")
            return False

        positions_to_remove = set(self.doc_index[doc_id])
        logger.info(
            f"Removing {len(positions_to_remove)} chunks for "
            f"doc_id={doc_id[:12]}..."
        )

        # Rebuild index without the removed positions
        new_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        new_metadata = {}
        new_doc_index: Dict[str, List[int]] = {}
        new_pos = 0

        # Reconstruct vectors one by one (IndexFlatIP supports reconstruct)
        for old_pos in range(self.index.ntotal):
            if old_pos in positions_to_remove:
                continue

            vector = self.index.reconstruct(old_pos).reshape(1, -1)
            new_index.add(vector)

            chunk = self.metadata[old_pos]
            new_metadata[new_pos] = chunk

            if chunk.doc_id not in new_doc_index:
                new_doc_index[chunk.doc_id] = []
            new_doc_index[chunk.doc_id].append(new_pos)

            new_pos += 1

        # Replace internal state
        self.index = new_index
        self.metadata = new_metadata
        self.doc_index = new_doc_index
        del self.doc_filenames[doc_id]

        logger.info(f"Index rebuilt: {self.index.ntotal} chunks remaining")
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[SearchResult]:
        """
        Search for the top-k most similar chunks to the query embedding.
        Returns SearchResult objects with chunk metadata and scores.
        """
        if self.index.ntotal == 0:
            return []

        # Clamp top_k to available chunks
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata.get(int(idx))
            if chunk:
                results.append(SearchResult(
                    chunk=chunk,
                    score=float(score),
                    global_index=int(idx)
                ))

        return results

    def get_document_info(self) -> Dict[str, dict]:
        """Return metadata about all indexed documents."""
        info = {}
        for doc_id, positions in self.doc_index.items():
            filename = self.doc_filenames.get(doc_id, "unknown")
            info[doc_id] = {
                "filename": filename,
                "file_hash": doc_id,
                "chunk_count": len(positions)
            }
        return info

    def has_document(self, doc_id: str) -> bool:
        """Check if a document is already indexed."""
        return doc_id in self.doc_index

    def get_doc_id_by_filename(self, filename: str) -> Optional[str]:
        """Find the doc_id for a given filename."""
        for doc_id, name in self.doc_filenames.items():
            if name == filename:
                return doc_id
        return None

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(INDEX_FILE))

        meta_data = {
            "metadata": self.metadata,
            "doc_index": self.doc_index,
            "doc_filenames": self.doc_filenames,
        }
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(meta_data, f)

        logger.info(
            f"Saved index ({self.index.ntotal} chunks, "
            f"{len(self.doc_index)} documents) to {INDEX_DIR}"
        )

    def load(self) -> bool:
        """Load the FAISS index and metadata from disk. Returns True if loaded."""
        if not INDEX_FILE.exists() or not METADATA_FILE.exists():
            logger.info("No existing index found on disk")
            return False

        try:
            self.index = faiss.read_index(str(INDEX_FILE))

            with open(METADATA_FILE, "rb") as f:
                meta_data = pickle.load(f)

            self.metadata = meta_data["metadata"]
            self.doc_index = meta_data["doc_index"]
            self.doc_filenames = meta_data["doc_filenames"]

            logger.info(
                f"Loaded index ({self.index.ntotal} chunks, "
                f"{len(self.doc_index)} documents) from disk"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
