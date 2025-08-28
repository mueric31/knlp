# chat.py
import os, json, sys, re
import faiss
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from config import (
    FAISS_PATH, META_PATH, SYN_PATH,
    EMBED_MODEL, CHAT_MODEL,
    TOP_K, SCORE_THRESHOLD
)

FALLBACK = "ntamakuru ndagira kuri iyi ngingo"


# ---------- I/O helpers ----------
def load_meta(meta_path: str) -> List[Dict]:
    rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def load_synonyms(path: str) -> Dict[str, List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # ensure values are lists of strings
            clean = {}
            for k, v in (data or {}).items():
                if isinstance(v, list):
                    clean[str(k)] = [str(x) for x in v]
            return clean
    except Exception:
        return {}


# ---------- Query processing ----------
def _clean_kiny_query(q: str) -> str:
    """
    Normalize common Kinyarwanda question fillers to improve retrieval.
    Examples: "ese", "none se", "mbese", "ni iki"/"niki"
    """
    ql = q.lower().strip()
    fillers = [
        r"^ese\s+",
        r"^none\s+se\s+",
        r"^none\s+",
        r"^mbese\s+",
        r"^ni\s+iki\s+",
        r"^niki\s+",
    ]
    for pat in fillers:
        ql = re.sub(pat, "", ql)
    return re.sub(r"\s+", " ", ql).strip()

def expand_query_with_synonyms(q: str, syn: Dict[str, List[str]]) -> str:
    """
    Expand query using synonyms:
    - If a key appears in the query, add all its synonyms
    - If any synonym appears in the query, add the key as well
    """
    if not syn:
        return q

    q_low = q.lower()
    additions = set()

    for key, candidates in syn.items():
        key_l = key.lower()
        # if key is in query => add its synonyms
        if key_l in q_low:
            for c in candidates:
                c = c.lower().strip()
                if c:
                    additions.add(c)
        # if any synonym is in query => add the key
        for c in candidates:
            c_l = c.lower().strip()
            if c_l and c_l in q_low:
                additions.add(key_l)

    if additions:
        return q + " | " + " ".join(sorted(additions))
    return q


# ---------- Embedding & retrieval ----------
def embed_query(client: OpenAI, text: str) -> np.ndarray:
    emb = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    x = np.array(emb, dtype="float32")
    faiss.normalize_L2(x.reshape(1, -1))
    return x

def _keyword_candidates(meta_rows: List[Dict], query: str, syn: Dict[str, List[str]], topn: int = 5):
    """
    Lightweight keyword fallback: rank chunks by how many query terms (and synonym terms) they contain.
    """
    q = _clean_kiny_query(query).lower()
    tokens = [t for t in re.split(r"[^\w’'’]+", q) if len(t) > 2]

    # consider synonyms for keys present in query
    extra = set()
    for key, vals in (syn or {}).items():
        key_l = key.lower()
        if key_l in q:
            for v in vals:
                v = str(v).lower().strip()
                if v and len(v) > 2:
                    extra.add(v)
        # also if any synonym appears, include the key
        for v in vals:
            v_l = str(v).lower().strip()
            if v_l and v_l in q and len(v_l) > 2:
                extra.add(key_l)

    vocab = set(tokens) | extra
    if not vocab:
        return []

    scored = []
    for row in meta_rows:
        text = str(row.get("text", "")).lower()
        score = sum(text.count(term) for term in vocab)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:topn]]

def retrieve(client: OpenAI, index, meta_rows: List[Dict], question: str, syn: Dict[str, List[str]]):
    base_q = _clean_kiny_query(question)
    qx = expand_query_with_synonyms(base_q, syn)
    qvec = embed_query(client, qx)

    scores, idxs = index.search(qvec.reshape(1, -1), TOP_K)
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    pairs = []
    for score, i in zip(scores, idxs):
        if i == -1:
            continue
        pairs.append((score, meta_rows[i]))

    # Filter by threshold
    good = [m for (s, m) in pairs if s >= SCORE_THRESHOLD]

    # Fallback: add up to 3 keyword-matched chunks if nothing passed threshold
    if not good:
        kw = _keyword_candidates(meta_rows, question, syn, topn=5)
        if kw:
            good = kw[:3]

    return good, scores[: len(good)]


# ---------- LLM answering ----------
def format_context(chunks: List[Dict]) -> str:
    out = []
    for i, ch in enumerate(chunks, 1):
        out.append(f"[Igice {i} | Page {ch.get('page','?')}] {ch['text']}")
    return "\n\n".join(out)

def ask_llm(client: OpenAI, context: str, question: str) -> str:
    system = (
        "Uri umufasha uvuga Kinyarwanda. Subiza ikibazo ukoresheje **amabwire aboneka gusa** mu CONTEXT. "
        f"NIBA CONTEXT idafite igisubizo, subiza gusa uti: '{FALLBACK}'. "
        "Irinde gukabya cyangwa guhanga ibisubizo bidashingiye ku nyandiko."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nIKIBAZO:\n{question}"},
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ---------- Main ----------
def ensure_index_loaded(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"FAISS index not found at {path}. Run build_index.py first.")
    return faiss.read_index(path)

def main():
    load_dotenv()

    # Use explicit API key if available for robustness on Windows
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    index = ensure_index_loaded(FAISS_PATH)
    meta_rows = load_meta(META_PATH)
    synonyms = load_synonyms(SYN_PATH)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        if not question:
            print(FALLBACK); return
        chunks, _ = retrieve(client, index, meta_rows, question, synonyms)
        if not chunks:
            print(FALLBACK); return
        context = format_context(chunks)
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
            chunks, _ = retrieve(client, index, meta_rows, question, synonyms)
            if not chunks:
                print(FALLBACK); continue
            context = format_context(chunks)
            answer = ask_llm(client, context, question)
            print(answer)

if __name__ == "__main__":
    main()
