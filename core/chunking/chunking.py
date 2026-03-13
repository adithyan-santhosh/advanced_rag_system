import re
from typing import List
import logging

def clean_text(text: str) -> str:
    """
    Cleans text while preserving paragraph boundaries.
    """
    # Normalize Windows line endings
    text = text.replace("\r\n", "\n")

    # Remove extra spaces but preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove excessive newlines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()



def fixed_size_chunking(text: str, chunk_size: int = 500) -> List[str]:
    """
    Splits text into fixed-size chunks.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def overlapping_chunking(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Splits text into overlapping chunks.
    Helps preserve context across chunk boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def paragraph_chunking(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Splits text by paragraph boundaries.
    If paragraph too long, splits further.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chunk_size:
            chunks.append(para)
        else:
            for i in range(0, len(para), max_chunk_size):
                chunks.append(para[i:i + max_chunk_size])

    return chunks

def semantic_chunking(text: str, max_chunk_size: int = 300) -> List[str]:
    """
    Paragraph-aware chunking with sentence-level fallback.
    Preserves semantic boundaries.
    """

    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chunk_size:
            chunks.append(para)
        else:
            # Sentence split fallback
            sentences = re.split(r'(?<=[.!?])\s+', para)

            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_chunk_size:
                    current_chunk += " " + sentence
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

    return chunks