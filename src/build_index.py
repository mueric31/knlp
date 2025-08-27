# Build the FAISS index from your source doc(s).
# Run on Render as a Build Command:
#   pip install -r requirements.txt && python -m src.build_index

import json
from typing import List, Dict

import faiss
import numpy as np
from openai import OpenAI

# *** IMPORTANT: relative imports inside the package ***
from .config import PDF_PATH, FAISS_PATH, META_PATH, EMBED_MODEL, CHUNK_SIZE, OVERLAP
from .utils import read_pdf_text, chunk_by_tokens

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    out = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in out.data]

def main():
    text = read_pdf_text(PDF_PATH)
    chunks = chunk_by_tokens(text, CHUNK_SIZE, OVERLAP)

    client = OpenAI()

    # Embed chunks
    vectors = embed_texts(client, chunks)
    vecs = np.array(vectors, dtype="float32")
    dim = vecs.shape[1]

    # Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    faiss.write_index(index, FAISS_PATH)

    # Write metadata (one JSON line per chunk)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(json.dumps({"id": i, "text": chunk}, ensure_ascii=False) + "\n")

    print(f"Built index: {FAISS_PATH}, meta: {META_PATH}, chunks: {len(chunks)}")

if __name__ == "__main__":
    main()
