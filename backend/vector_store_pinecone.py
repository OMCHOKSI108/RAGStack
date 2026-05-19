"""
Pinecone-backed vector store with the same interface as the FAISS VectorStore.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from backend.config import (
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_HOST,
    PINECONE_INDEX,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
)
from backend.models import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """Pinecone-backed vector store for Render-friendly persistence."""

    def __init__(self):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is required when VECTOR_STORE=PINECONE")
        if not PINECONE_INDEX:
            raise RuntimeError("PINECONE_INDEX is required when VECTOR_STORE=PINECONE")

        from pinecone import Pinecone, ServerlessSpec

        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        index_list = self.pc.list_indexes()
        if hasattr(index_list, "names"):
            existing = set(index_list.names())
        else:
            existing = {idx["name"] if isinstance(idx, dict) else idx.name for idx in index_list}
        if PINECONE_INDEX not in existing:
            logger.info("Creating Pinecone index '%s'", PINECONE_INDEX)
            self.pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
            while True:
                description = self.pc.describe_index(PINECONE_INDEX)
                status = description.get("status", {}) if isinstance(description, dict) else description.status
                ready = status.get("ready", False) if isinstance(status, dict) else status.ready
                if ready:
                    break
                time.sleep(1)

        if PINECONE_HOST:
            self.index = self.pc.Index(host=PINECONE_HOST)
        else:
            self.index = self.pc.Index(PINECONE_INDEX)

    @staticmethod
    def _vector_id(doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}:{chunk_index}"

    @property
    def total_chunks(self) -> int:
        stats = self.index.describe_index_stats()
        namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else getattr(stats, "namespaces", {})
        namespace = namespaces.get(PINECONE_NAMESPACE, {})
        if isinstance(namespace, dict):
            return int(namespace.get("vector_count", 0))
        return int(getattr(namespace, "vector_count", 0))

    @property
    def total_documents(self) -> int:
        return len(self.get_document_info())

    @property
    def doc_index(self) -> dict[str, list[int]]:
        return {
            doc_id: list(range(info["chunk_count"]))
            for doc_id, info in self.get_document_info().items()
        }

    def _metadata_scan(self, filter: Optional[dict] = None, top_k: int = 10000) -> list[dict]:
        if self.total_chunks == 0:
            return []

        response = self.index.query(
            namespace=PINECONE_NAMESPACE,
            vector=[0.0] * EMBEDDING_DIMENSION,
            top_k=min(max(self.total_chunks, 1), top_k),
            include_metadata=True,
            filter=filter,
        )
        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
        return [match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {}) for match in matches]

    def add(self, embeddings: np.ndarray, chunks: list[DocumentChunk], doc_id: str) -> None:
        """Upsert document chunks and embeddings to Pinecone."""
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length")

        vectors = []
        for embedding, chunk in zip(embeddings, chunks, strict=False):
            vectors.append(
                {
                    "id": self._vector_id(doc_id, chunk.chunk_index),
                    "values": embedding.astype(float).tolist(),
                    "metadata": {
                        "doc_id": doc_id,
                        "filename": chunk.source_file,
                        "source_file": chunk.source_file,
                        "page_number": int(chunk.page_number),
                        "chunk_index": int(chunk.chunk_index),
                        "text": chunk.text,
                    },
                }
            )

        for start in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[start:start + 100], namespace=PINECONE_NAMESPACE)

        logger.info("Upserted %s chunks for doc_id=%s", len(chunks), doc_id[:12])

    def remove_document(self, doc_id: str) -> bool:
        """Delete all chunks belonging to a document."""
        if not self.has_document(doc_id):
            logger.warning("Document %s not found in Pinecone", doc_id[:12])
            return False
        self.index.delete(namespace=PINECONE_NAMESPACE, filter={"doc_id": {"$eq": doc_id}})
        logger.info("Deleted Pinecone vectors for doc_id=%s", doc_id[:12])
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[SearchResult]:
        """Search Pinecone and return SearchResult objects."""
        if self.total_chunks == 0:
            return []

        vector = query_embedding.reshape(-1).astype(float).tolist()
        response = self.index.query(
            namespace=PINECONE_NAMESPACE,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])

        results = []
        for i, match in enumerate(matches):
            metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            score = match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)
            chunk = DocumentChunk(
                text=str(metadata.get("text", "")),
                source_file=str(metadata.get("source_file") or metadata.get("filename") or "unknown"),
                page_number=int(metadata.get("page_number", 0)),
                chunk_index=int(metadata.get("chunk_index", i)),
                doc_id=str(metadata.get("doc_id", "")),
            )
            results.append(SearchResult(chunk=chunk, score=float(score), global_index=i))
        return results

    def get_document_info(self) -> dict[str, dict]:
        """Return metadata about all indexed documents."""
        info: dict[str, dict] = {}
        for metadata in self._metadata_scan():
            doc_id = str(metadata.get("doc_id", ""))
            if not doc_id:
                continue
            if doc_id not in info:
                info[doc_id] = {
                    "filename": metadata.get("filename") or metadata.get("source_file") or "unknown",
                    "file_hash": doc_id,
                    "chunk_count": 0,
                }
            info[doc_id]["chunk_count"] += 1
        return info

    def has_document(self, doc_id: str) -> bool:
        """Check if a document is already indexed."""
        return bool(self._metadata_scan(filter={"doc_id": {"$eq": doc_id}}, top_k=1))

    def get_doc_id_by_filename(self, filename: str) -> Optional[str]:
        """Find a doc_id by filename."""
        matches = self._metadata_scan(filter={"filename": {"$eq": filename}}, top_k=1)
        if not matches:
            matches = self._metadata_scan(filter={"source_file": {"$eq": filename}}, top_k=1)
        if not matches:
            return None
        return str(matches[0].get("doc_id"))

    def save(self) -> None:
        """No-op: Pinecone persists vectors externally."""
        return None

    def load(self) -> bool:
        """No-op: Pinecone index is remote and already persistent."""
        return True
