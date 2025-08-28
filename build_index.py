import json, os, faiss, numpy as np
from tqdm import tqdm
from openai import OpenAI
from config import PDF_PATH, FAISS_PATH, META_PATH, EMBED_MODEL, CHUNK_SIZE, OVERLAP
from utils import read_pdf_text, chunk_by_tokens

def embed_texts(client: OpenAI, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [e.embedding for e in resp.data]

def main():
    print(f"Reading PDF from: {PDF_PATH}")
    pages = read_pdf_text(PDF_PATH)
    all_chunks = []
    meta = []
    for page_no, txt in pages:
        if not txt.strip():
            continue
        chunks = chunk_by_tokens(txt, CHUNK_SIZE, OVERLAP)
        for ch in chunks:
            all_chunks.append(ch)
            meta.append({"page": page_no, "text": ch})

    if not all_chunks:
        raise RuntimeError("No text found in PDF.")

    print(f"Embedding {len(all_chunks)} chunks with model {EMBED_MODEL} ...")
    client = OpenAI()
    embs = []
    B = 64
    for i in tqdm(range(0, len(all_chunks), B)):
        batch = all_chunks[i : i+B]
        embs.extend(embed_texts(client, batch))

    X = np.array(embs).astype("float32")
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(X)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved index to {FAISS_PATH} and meta to {META_PATH}")

if __name__ == "__main__":
    main()
