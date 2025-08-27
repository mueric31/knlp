import re
from typing import List, Tuple
from pypdf import PdfReader
import tiktoken

def read_pdf_text(path: str) -> List[Tuple[int, str]]:
    """
    Returns a list of (page_number, page_text) tuples.
    """
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        # Basic cleanup
        txt = re.sub(r"\s+\n", "\n", txt).strip()
        pages.append((i + 1, txt))
    return pages

def split_into_sentences(text: str) -> List[str]:
    # Simple sentence splitter for Kinyarwanda text (periods, question marks, exclamation marks, ellipses)
    parts = re.split(r"(?<=[\.\?\!…])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_by_tokens(text: str, chunk_size: int = 900, overlap: int = 200, encoding_name: str = "cl100k_base") -> List[str]:
    """
    Token-based chunking with overlaps to keep context continuity.
    """
    enc = tiktoken.get_encoding(encoding_name)
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        window = toks[i : i + chunk_size]
        chunks.append(enc.decode(window))
        i += (chunk_size - overlap) if (chunk_size - overlap) > 0 else chunk_size
    return chunks
