# chat.py
import os, json, sys, re
import faiss
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# --- Robust import of config (same folder OR parent folder) ---
try:
    from config import (
        FAISS_PATH, META_PATH, SYN_PATH,
        EMBED_MODEL, CHAT_MODEL,
        TOP_K, SCORE_THRESHOLD,
    )
except ImportError:
    from pathlib import Path
    parent = Path(__file__).resolve().parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from config import (
        FAISS_PATH, META_PATH, SYN_PATH,
        EMBED_MODEL, CHAT_MODEL,
        TOP_K, SCORE_THRESHOLD,
    )

# Strict fallback phrase (as requested)
FALLBACK = "nta makuru nari nabona"


# ---------- I/O helpers ----------
def load_meta(meta_path: str) -> List[Dict]:
    rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def load_synonyms(path: str) -> Dict[str, List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
            clean = {}
            for k, v in data.items():
                if isinstance(v, list):
                    clean[str(k)] = [str(x) for x in v]
            return clean
    except Exception:
        return {}


# ---------- Query processing ----------
def _clean_kiny_query(q: str) -> str:
    """
    Normalize common Kinyarwanda fillers to improve retrieval.
    """
    ql = q.lower().strip()
    fillers = [
        r"^ese\s+", r"^none\s+se\s+", r"^none\s+",
        r"^mbese\s+", r"^ni\s+iki\s+", r"^niki\s+",
    ]
    for pat in fillers:
        ql = re.sub(pat, "", ql)
    return re.sub(r"\s+", " ", ql).strip()

def expand_query_with_synonyms(q: str, syn: Dict[str, List[str]]) -> str:
    """
    Expand query using synonyms:
    - If a key appears in the query, add its synonyms
    - If any synonym appears, add the key
    """
    if not syn:
        return q
    q_low = q.lower()
    additions = set()
    for key, candidates in syn.items():
        key_l = key.lower()
        if key_l in q_low:
            for c in candidates:
                c = c.lower().strip()
                if c:
                    additions.add(c)
        for c in candidates:
            c_l = c.lower().strip()
            if c_l and c_l in q_low:
                additions.add(key_l)
    return q if not additions else (q + " " + " ".join(sorted(additions)))


# ---------- Embedding & retrieval ----------
def embed_query(client: OpenAI, text: str) -> np.ndarray:
    emb = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    x = np.array(emb, dtype="float32")
    # Normalize query; index may be L2/IP — we will gate by lexical match anyway.
    faiss.normalize_L2(x.reshape(1, -1))
    return x

def _keyword_overlap_score(text: str, vocab: set) -> int:
    t = text.lower()
    # simple count of term occurrences (robust for Kinyarwanda)
    return sum(t.count(term) for term in vocab)

def _keyword_candidates(meta_rows: List[Dict], query: str, syn: Dict[str, List[str]], topn: int = 5):
    """
    Lightweight keyword fallback: rank chunks by how many query terms (and synonym terms) they contain.
    """
    q = _clean_kiny_query(query).lower()
    tokens = [t for t in re.split(r"[^\w’'’]+", q) if len(t) > 2]

    # consider synonyms for keys present in query, and vice versa
    extra = set()
    for key, vals in (syn or {}).items():
        key_l = key.lower()
        if key_l in q:
            for v in vals:
                v = str(v).lower().strip()
                if v and len(v) > 2:
                    extra.add(v)
        for v in vals:
            v_l = str(v).lower().strip()
            if v_l and v_l in q and len(v_l) > 2:
                extra.add(key_l)

    vocab = set(tokens) | extra
    if not vocab:
        return []

    scored = []
    for row in meta_rows:
        text = str(row.get("text", ""))
        score = _keyword_overlap_score(text, vocab)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:topn]]

def retrieve(client: OpenAI, index, meta_rows: List[Dict], question: str, syn: Dict[str, List[str]]):
    """
    Vector search + lexical gating to guarantee 'book-only' answers.
    If nothing relevant is found, return [] so caller prints FALLBACK.
    """
    base_q = _clean_kiny_query(question)
    qx = expand_query_with_synonyms(base_q, syn)
    qvec = embed_query(client, qx)

    # Vector search (TOP_K from config)
    D, I = index.search(qvec.reshape(1, -1), max(1, int(TOP_K)))
    D = D[0].tolist(); I = I[0].tolist()

    # Candidate chunks from vector search
    vec_chunks = [meta_rows[i] for i in I if 0 <= i < len(meta_rows)]

    # Lexical relevance gate: ensure chunks actually mention terms from the question
    vocab = set([t for t in re.split(r"[^\w’'’]+", base_q.lower()) if len(t) > 2])
    good = [ch for ch in vec_chunks if _keyword_overlap_score(ch.get("text", ""), vocab) > 0]

    if not good:
        # Fallback: try keyword-based candidates
        kw = _keyword_candidates(meta_rows, question, syn, topn=5)
        if kw:
            good = kw[:3]

    # Final guarantee: if still nothing, return []
    return good


# ---------- LLM answering ----------
def format_context(chunks: List[Dict]) -> str:
    out = []
    for i, ch in enumerate(chunks, 1):
        page = ch.get("page", "?")
        text = ch.get("text", "").strip()
        if not text:
            continue
        out.append(f"[Igice {i} | Page {page}]\n{text}")
    return "\n\n---\n\n".join(out).strip()

def ask_llm(client: OpenAI, context: str, question: str) -> str:
    system = (
        "Uri umufasha uvuga Kinyarwanda.\n"
        "Subiza ikibazo ukoresheje **amabwire aboneka gusa** mu CONTEXT iri hepfo.\n"
        f"NIBA CONTEXT idafite igisubizo, subiza gusa uti: '{FALLBACK}'.\n"
        "Ntugire ibyo wihangira cyangwa wongeraho ibitagaragara muri CONTEXT."
    )
    user = f"CONTEXT:\n{context}\n\nIKIBAZO:\n{question}\n\nSUBIZA MU NTERURO NKEYA:"

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt if txt else FALLBACK
    except Exception:
        return FALLBACK


# ---------- Main ----------
def ensure_index_loaded(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"FAISS index not found at {path}. Run build_index.py first.")
    return faiss.read_index(path)

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    index = ensure_index_loaded(FAISS_PATH)
    meta_rows = load_meta(META_PATH)
    synonyms = load_synonyms(SYN_PATH)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        if not question:
            print(FALLBACK); return
        chunks = retrieve(client, index, meta_rows, question, synonyms)
        if not chunks:
            print(FALLBACK); return
        context = format_context(chunks)
        if not context:
            print(FALLBACK); return
        answer = ask_llm(client, context, question)
        print(answer)
    else:
        print("Andika ikibazo cyawe (Ctrl+C gusohoka):")
        while True:
            try:
                question = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nMurakoze!")
                break
            if not question:
                print(FALLBACK); continue
            chunks = retrieve(client, index, meta_rows, question, synonyms)
            if not chunks:
                print(FALLBACK); continue
            context = format_context(chunks)
            if not context:
                print(FALLBACK); continue
            answer = ask_llm(client, context, question)
            print(answer)

if __name__ == "__main__":
    main()
