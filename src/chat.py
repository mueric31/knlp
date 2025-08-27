import os
import json
from functools import lru_cache
from typing import List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# *** IMPORTANT: use relative imports inside the package ***
from .config import (
    FAISS_PATH, META_PATH, SYN_PATH,  # SYN_PATH optional
    EMBED_MODEL, CHAT_MODEL,
    TOP_K, SCORE_THRESHOLD,
)

FALLBACK = "Ntabisubizo bubonetse."  # or "No answer available."

# ---------- Bootstrapping ----------

@lru_cache()
def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else OpenAI()

def _file_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)

@lru_cache()
def load_index_and_meta() -> Tuple[faiss.Index, list]:
    index = None
    meta_rows = []
    if _file_exists(FAISS_PATH) and _file_exists(META_PATH):
        index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta_rows = [json.loads(line) for line in f]
    return index, meta_rows

@lru_cache()
def load_synonyms() -> dict:
    if _file_exists(SYN_PATH):
        try:
            with open(SYN_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

# ---------- Embeddings / Search ----------

def embed_texts(texts: List[str]) -> List[List[float]]:
    client = get_client()
    out = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in out.data]

def search_index(query_vec: List[float], index: faiss.Index, k: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array([query_vec], dtype="float32")
    D, I = index.search(arr, k)
    return D[0], I[0]

# ---------- Prompting ----------

def format_context(meta_rows: List[dict], idxs: np.ndarray, limit_chars: int = 4000) -> str:
    parts = []
    for i in idxs:
        if 0 <= int(i) < len(meta_rows):
            m = meta_rows[int(i)]
            # support various meta schemas
            text = m.get("text") or m.get("chunk") or m.get("content") or ""
            if text:
                parts.append(text)
    context = "\n\n---\n\n".join(parts)
    return context[:limit_chars]

def ask_llm(question: str, context: str = "") -> str:
    client = get_client()
    system = (
        "You are a helpful assistant. Use the provided context if it is relevant. "
        "If the context is empty or irrelevant, answer from your general knowledge. "
        "If the answer is truly unknown, say you don't know."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"{FALLBACK} ({e})"

# ---------- Public API used by FastAPI ----------

def get_response(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return FALLBACK

    index, meta_rows = load_index_and_meta()

    # If index is missing, just answer directly.
    if not index or not meta_rows:
        return ask_llm(q, context="")

    # (Optional) expand with synonyms
    syns = load_synonyms()
    expansions = []
    for word, alts in syns.items():
        if word.lower() in q.lower():
            expansions.extend(alts)
    expanded_q = q if not expansions else (q + " " + " ".join(expansions))

    # Embed, search, and build context
    q_vec = embed_texts([expanded_q])[0]
    k = min(max(int(TOP_K), 1), len(meta_rows)) if isinstance(TOP_K, int) else min(5, len(meta_rows))
    D, I = search_index(q_vec, index, k)

    # Filter by threshold if provided; note FAISS L2 distance: smaller = closer
    # If SCORE_THRESHOLD is a similarity threshold in [0..1], you may convert distances to similarities.
    # Here we simply keep all top-k; adjust if you rely on a real thresholding scheme.
    context = format_context(meta_rows, I)

    return ask_llm(q, context=context)
