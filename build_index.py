# build_index.py
import os, json, re
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# --- Imports that work both as a package and as a script ---
try:
    # When run as a module:  python -m src.build_index
    from .config import PDF_PATH, FAISS_PATH, META_PATH, EMBED_MODEL, CHUNK_SIZE, OVERLAP
except ImportError:
    # When run directly in src/:  python build_index.py
    from config import PDF_PATH, FAISS_PATH, META_PATH, EMBED_MODEL, CHUNK_SIZE, OVERLAP

load_dotenv()
client = OpenAI()

WS_RE = re.compile(r"\s+")

def read_pdf(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    reader = PdfReader(path)
    texts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            # normalize whitespace early
            texts.append(WS_RE.sub(" ", txt).strip())
    doc = "\n\n".join(texts).strip()
    if not doc:
        raise ValueError("No text could be extracted from the PDF.")
    return doc

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    n = len(tokens)
    step = max(size - overlap, 1)
    while i < n:
        piece = tokens[i:i+size]
        if not piece:
            break
        chunks.append(" ".join(piece).strip())
        i += step
    # dedupe consecutive duplicates (rare, but safe)
    deduped = []
    prev = None
    for c in chunks:
        if c != prev:
            deduped.append(c)
        prev = c
    return deduped

def embed_batch(texts: List[str]) -> np.ndarray:
    # OpenAI embeddings (v3)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    arr = np.array(vecs, dtype="float32")
    return arr

def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def write_meta(meta_path: str, chunks: List[str]) -> None:
    out = Path(meta_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")

def main():
    print(f"PDF_PATH   = {PDF_PATH}")
    print(f"FAISS_PATH = {FAISS_PATH}")
    print(f"META_PATH  = {META_PATH}")
    print(f"EMBED_MODEL= {EMBED_MODEL}")
    print(f"CHUNK_SIZE = {CHUNK_SIZE}, OVERLAP = {OVERLAP}")

    doc = read_pdf(PDF_PATH)
    print(f"✅ Extracted ~{len(doc):,} chars from PDF")

    chunks = chunk_text(doc, CHUNK_SIZE, OVERLAP)
    if not chunks:
        raise ValueError("No chunks created; check CHUNK_SIZE/OVERLAP.")
    print(f"✅ Created {len(chunks):,} chunks")

    # embed in reasonable batches to avoid payload limits
    all_vecs: List[np.ndarray] = []
    B = 64
    for i in range(0, len(chunks), B):
        batch = chunks[i:i+B]
        vecs = embed_batch(batch)
        all_vecs.append(vecs)
        print(f"… embedded {min(i+B, len(chunks))}/{len(chunks)}")

    embeddings = np.vstack(all_vecs)
    print(f"✅ Got embeddings shape: {embeddings.shape}")

    index = build_faiss(embeddings)
    faiss.write_index(index, FAISS_PATH)
    print(f"💾 Wrote FAISS index -> {FAISS_PATH}")

    write_meta(META_PATH, chunks)
    print(f"💾 Wrote meta jsonl -> {META_PATH}")

    print("🎉 Done.")

if __name__ == "__main__":
    main()
