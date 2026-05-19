"""
Document parser for PDF, Markdown, and plain text files.
Extracts structured sections with source metadata.
Preserves UTF-8 currency symbols and normalizes common OCR artifacts.
"""

import logging
import re
from pathlib import Path
from typing import List

import pymupdf  # PyMuPDF (v1.27+ uses 'pymupdf' instead of 'fitz')
import markdown

from backend.config import OCR_CORRECTIONS_ENABLED
from backend.models import DocumentSection

logger = logging.getLogger(__name__)

# OCR artifact patterns: I2k → ₹12k, l500 → ₹500, etc.
_OCR_CURRENCY_PATTERNS = [
    (re.compile(r'\bI(\d+[kK]?)\b'), r'₹\1'),   # I12k → ₹12k
    (re.compile(r'\bl(\d+[kK]?)\b'), r'₹\1'),   # l500 → ₹500
    (re.compile(r'\b1(\d{3,})\b'), r'₹\1'),     # 15000 → ₹15000 (ambiguous, only 4+ digits)
    (re.compile(r'Rs\.?\s*(\d+)'), r'₹\1'),     # Rs. 5000 → ₹5000
    (re.compile(r'INR\s*(\d+)'), r'₹\1'),       # INR 5000 → ₹5000
]


def normalize_ocr_artifacts(text: str) -> str:
    """
    Normalize common OCR artifacts in extracted text.
    Only runs when OCR_CORRECTIONS_ENABLED is set; the default-off behavior
    avoids mangling technical text like "l1 regularization" or "1500 params".
    Returns (cleaned_text, had_corrections).
    """
    if not OCR_CORRECTIONS_ENABLED:
        return text, False
    had_corrections = False
    for pattern, replacement in _OCR_CURRENCY_PATTERNS:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
            had_corrections = True
    return text, had_corrections


def parse_pdf(file_path: Path) -> List[DocumentSection]:
    """
    Extract text from a PDF file, one section per page.
    Uses PyMuPDF for fast, reliable UTF-8 text extraction.
    Normalizes OCR artifacts after extraction.
    """
    sections = []
    total_corrections = 0
    try:
        doc = pymupdf.open(str(file_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract as UTF-8 text (PyMuPDF returns UTF-8 by default)
            text = page.get_text("text", sort=True).strip()
            if text:
                # Normalize OCR artifacts
                cleaned, had_corrections = normalize_ocr_artifacts(text)
                if had_corrections:
                    total_corrections += 1
                sections.append(DocumentSection(
                    text=cleaned,
                    source_file=file_path.name,
                    page_or_section=page_num + 1,
                    raw_text=text if cleaned != text else None,
                ))
        doc.close()
    except Exception as e:
        logger.error(f"Failed to parse PDF {file_path.name}: {e}")
        raise

    logger.info(
        f"Parsed PDF '{file_path.name}': {len(sections)} pages extracted, "
        f"{total_corrections} pages with OCR corrections"
    )
    return sections


def parse_markdown(file_path: Path) -> List[DocumentSection]:
    """
    Parse a Markdown file, splitting on heading boundaries.
    Preserves heading context for each section.
    """
    sections = []
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = file_path.read_text(encoding="latin-1")

    # Normalize OCR artifacts
    cleaned, _ = normalize_ocr_artifacts(raw_text)

    # Split on heading markers to preserve structure
    lines = cleaned.split("\n")
    current_section_lines = []
    section_number = 1

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and current_section_lines:
            section_text = "\n".join(current_section_lines).strip()
            if section_text:
                sections.append(DocumentSection(
                    text=section_text,
                    source_file=file_path.name,
                    page_or_section=section_number
                ))
                section_number += 1
            current_section_lines = [line]
        else:
            current_section_lines.append(line)

    if current_section_lines:
        section_text = "\n".join(current_section_lines).strip()
        if section_text:
            sections.append(DocumentSection(
                text=section_text,
                source_file=file_path.name,
                page_or_section=section_number
            ))

    logger.info(f"Parsed Markdown '{file_path.name}': {len(sections)} sections extracted")
    return sections


def parse_text(file_path: Path) -> List[DocumentSection]:
    """
    Parse a plain text file, splitting on double newlines (paragraphs).
    """
    sections = []
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = file_path.read_text(encoding="latin-1")

    # Normalize OCR artifacts
    cleaned, _ = normalize_ocr_artifacts(raw_text)

    paragraphs = cleaned.split("\n\n")
    section_number = 1

    for para in paragraphs:
        text = para.strip()
        if text:
            sections.append(DocumentSection(
                text=text,
                source_file=file_path.name,
                page_or_section=section_number
            ))
            section_number += 1

    logger.info(f"Parsed text '{file_path.name}': {len(sections)} sections extracted")
    return sections


def parse_document(file_path: Path) -> List[DocumentSection]:
    """
    Dispatch to the appropriate parser based on file extension.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix == ".md":
        return parse_markdown(file_path)
    elif suffix == ".txt":
        return parse_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
