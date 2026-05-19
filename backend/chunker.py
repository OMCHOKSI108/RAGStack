"""
Semantic text chunker using recursive character splitting.
Splits by paragraph -> newline -> sentence -> word boundaries,
preserving meaning and heading context.
"""

import logging

from backend.config import CHUNK_OVERLAP, CHUNK_SEPARATORS, CHUNK_SIZE
from backend.models import DocumentChunk, DocumentSection

logger = logging.getLogger(__name__)


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """
    Recursively split text using a hierarchy of separators.
    Tries the most semantically meaningful separator first,
    then falls back to less meaningful ones.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the first separator that exists in the text
    chosen_separator = separators[-1]  # fallback to single char
    for sep in separators:
        if sep in text:
            chosen_separator = sep
            break

    parts = text.split(chosen_separator)
    chunks = []
    current_chunk = ""

    for part in parts:
        candidate = current_chunk + chosen_separator + part if current_chunk else part

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # If this single part exceeds chunk_size, recurse with finer separators
            if len(part) > chunk_size:
                remaining_separators = separators[separators.index(chosen_separator) + 1:]
                if remaining_separators:
                    sub_chunks = _recursive_split(part, remaining_separators, chunk_size)
                    chunks.extend(sub_chunks)
                else:
                    # Last resort: hard split
                    for i in range(0, len(part), chunk_size):
                        piece = part[i:i + chunk_size].strip()
                        if piece:
                            chunks.append(piece)
                current_chunk = ""
            else:
                current_chunk = part

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """
    Apply overlap between consecutive chunks by prepending
    the tail of the previous chunk to the current one.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        # Avoid duplicating if the chunk already starts with the overlap
        if not chunks[i].startswith(prev_tail):
            overlapped.append(prev_tail + " " + chunks[i])
        else:
            overlapped.append(chunks[i])

    return overlapped


def chunk_sections(
    sections: list[DocumentSection],
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: list[str] = None
) -> list[DocumentChunk]:
    """
    Split document sections into semantically meaningful chunks.
    Each chunk retains metadata about its source document, page, and position.
    """
    if separators is None:
        separators = CHUNK_SEPARATORS

    all_chunks = []
    chunk_index = 0

    for section in sections:
        # Split the section text recursively
        raw_chunks = _recursive_split(section.text, separators, chunk_size)

        # Apply overlap between consecutive chunks
        overlapped_chunks = _apply_overlap(raw_chunks, chunk_overlap)

        for chunk_text in overlapped_chunks:
            if not chunk_text.strip():
                continue

            all_chunks.append(DocumentChunk(
                text=chunk_text,
                source_file=section.source_file,
                page_number=section.page_or_section,
                chunk_index=chunk_index,
                doc_id=doc_id
            ))
            chunk_index += 1

    logger.info(
        f"Chunked '{sections[0].source_file if sections else 'unknown'}': "
        f"{len(sections)} sections -> {len(all_chunks)} chunks"
    )
    return all_chunks
