# src/chatbot.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, pickle, re
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch, tiktoken
from sklearn.metrics.pairwise import cosine_similarity

# IMPORTANT: relative import — gpt_model.py must be in src/
from .gpt_model import GPTConfig, GPTLanguageModel

FALLBACK = "nta makuru ahagije nari nagira"
INSTRUCTIONS = (
    "Subiza mu nteruro nkeya GUKORESHA gusa amakuru ari muri CONTEXT.\n"
    "NIBA CONTEXT idafite igisubizo, andika: 'nta makuru ahagije nari nagira'."
)

SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!…])\s+|\n+")
ALLOW_SINGLE_WORD = {"ikawa", "inzoga", "itabi", "ibiyobyabwenge"}

# ---------- lazy globals (loaded once) ----------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_enc = None
_model = None
_cfg = None
_idx = None
_sections_map: Dict[str, Dict[str, str]] = {}
_synonyms: Dict[str, List[str]] = {}

# thresholds & knobs (env-overridable)
_HI = float(os.getenv("HI", "0.40"))
_MID = float(os.getenv("MID", "0.28"))
_CONTEXT_TOPK = int(os.getenv("CONTEXT_TOPK", "2"))
_TITLE_BOOST = float(os.getenv("TITLE_BOOST", "1.25"))
_HIGH_MODE = os.getenv("HIGH_MODE", "section_sentences")  # sentence|section|section_sentences
_N_SENTENCES = int(os.getenv("N_SENTENCES", "0"))  # 0=all
_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "60"))
_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
_TOP_K = int(os.getenv("TOP_K", "1"))
_TOP_P = float(os.getenv("TOP_P", "1.0"))
_REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.03"))
_FREQ_PENALTY = float(os.getenv("FREQ_PENALTY", "0.3"))

def _first_sentence(txt: str) -> str:
    return re.split(r"(?<=[\.\?\!])\s", (txt or "").strip(), maxsplit=1)[0].strip() if txt else ""

def _split_sentences(text: str):
    return [s.strip() for s in SENT_SPLIT_RE.split(text or "") if s and s.strip()]

def _load_index(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    for k in ("vectorizer", "X", "items"):
        if k not in data:
            raise ValueError("Bad index file; rebuild with build_index_sentences.py")
    return data

def _load_sections_map(path: Path):
    if not path.exists(): return {}
    arr = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(s.get("id", "")).strip(): {
            "title": (s.get("title") or "").strip(),
            "content": (s.get("content") or "").strip()
        }
        for s in arr
    }

def _load_synonyms(path: Path|None) -> Dict[str, List[str]]:
    if not path or not path.exists(): return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for k, v in raw.items():
        key = str(k).lower().strip()
        out[key] = [str(x).lower().strip() for x in (v or []) if str(x).strip()]
    return out

def _expand_query(q: str, synonyms: Dict[str, List[str]]):
    q_low = (q or "").lower()
    matched, addon_terms = [], []
    for key, vars_ in synonyms.items():
        is_phrase = (" " in key) or ("’" in key) or ("'" in key)
        is_allowed_single = key in ALLOW_SINGLE_WORD or len(key) >= 6
        if is_phrase or is_allowed_single:
            if key in q_low or any(v in q_low for v in vars_):
                matched.append(key)
                addon_terms.extend([key] + vars_)
    expanded = (q or "")
    if addon_terms:
        expanded += " || " + " ".join(sorted(set(addon_terms)))
    return expanded, matched

def _retrieve(expanded_q, idx, matched_keys, synonyms, title_boost=1.25, topk=3):
    vec, X, items = idx["vectorizer"], idx["X"], idx["items"]
    sims = cosine_similarity(vec.transform([expanded_q]), X)[0]
    if matched_keys:
        sims = sims.copy()
        for i, it in enumerate(items):
            ttl = (it.get("title") or "").lower()
            hits = 0
            for k in matched_keys:
                cand = [k] + synonyms.get(k, [])
                if any(c in ttl for c in cand):
                    hits += 1
            if hits > 0:
                sims[i] *= (title_boost ** hits)
    order = np.argsort(sims)[::-1][:max(1, topk)]
    picks = [{"score": float(sims[i]), **items[i]} for i in order]
    return picks

def _build_context(picks):
    lines = []
    for p in picks:
        sec = p.get("section_id","?")
        ttl = (p.get("title") or "").strip()
        sent = (p.get("sentence") or "").strip()
        lines.append(f"- [{sec}] {ttl}: {sent}")
    return "\n".join(lines)

def _ensure_loaded():
    global _enc, _model, _cfg, _idx, _sections_map, _synonyms
    if _model is not None and _idx is not None and _enc is not None:
        return

    # Resolve paths relative to repo root (src/..)
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    ckpt = Path(os.getenv("CKPT", repo_root / "model/checkpoint/model.pt"))
    cfgp = Path(os.getenv("CFG",  repo_root / "model/checkpoint/config.json"))
    idxp = Path(os.getenv("INDEX", repo_root / "data/tfidf_sent.pkl"))
    secp = Path(os.getenv("SECTIONS", repo_root / "data/sections.json"))
    synp = Path(os.getenv("SYNONYMS", repo_root / "config/synonyms.json"))

    # Tokenizer
    _enc = tiktoken.get_encoding("gpt2")

    # Model
    cfg = GPTConfig(**json.loads(Path(cfgp).read_text(encoding="utf-8")))
    model = GPTLanguageModel(cfg).to(_DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=_DEVICE))
    model.eval()

    # Index + aux
    idx = _load_index(Path(idxp))
    sections_map = _load_sections_map(Path(secp))
    synonyms = _load_synonyms(Path(synp))

    _cfg = cfg
    _model = model
    _idx = idx
    _sections_map = sections_map
    _synonyms = synonyms

def get_response(q: str, show_source: bool = False) -> str:
    """
    Returns a short answer strictly from CONTEXT; falls back if low confidence.
    """
    _ensure_loaded()
    if not q or not q.strip():
        return FALLBACK

    expanded_q, matched_keys = _expand_query(q, _synonyms)
    picks = _retrieve(expanded_q, _idx, matched_keys, _synonyms,
                      title_boost=_TITLE_BOOST, topk=_CONTEXT_TOPK)
    if not picks:
        return FALLBACK

    best = picks[0]
    score = best["score"]
    sec = best.get("section_id","?")
    ttl = (best.get("title") or "").strip()
    sent = (best.get("sentence") or "").strip()

    # High confidence → extractive
    if score >= _HI:
        if _HIGH_MODE == "sentence":
            return sent or FALLBACK
        elif _HIGH_MODE == "section":
            sec_data = _sections_map.get(str(sec))
            if sec_data and sec_data.get("content"):
                return f"{sec_data['title']}\n{sec_data['content']}".strip() or FALLBACK
            return sent or FALLBACK
        else:  # section_sentences
            sec_data = _sections_map.get(str(sec))
            if sec_data and sec_data.get("content"):
                sents = _split_sentences(sec_data["content"])
                if _N_SENTENCES > 0:
                    sents = sents[:_N_SENTENCES]
                return (" ".join(sents)).strip() or FALLBACK
            return sent or FALLBACK

    # Medium confidence → strict RAG (generate from CONTEXT only)
    if score >= _MID:
        ctx = _build_context(picks)
        prompt = f"{INSTRUCTIONS}\n\nQ: {q}\nCONTEXT:\n{ctx}\nA:"
        ids = _enc.encode(prompt) or _enc.encode(" ")
        x = torch.tensor([ids], dtype=torch.long, device=_DEVICE)
        if x.size(1) > _cfg.block_size:
            x = x[:, -_cfg.block_size:]
        with torch.no_grad():
            out = _model.generate(
                x,
                max_new_tokens=_MAX_NEW_TOKENS,
                temperature=_TEMPERATURE,
                top_k=_TOP_K,
                top_p=_TOP_P,
                repeat_penalty=_REPEAT_PENALTY,
                freq_penalty=_FREQ_PENALTY,
            )
        text = _enc.decode(out[0].tolist())
        ans = _first_sentence(text.split("A:", 1)[-1].strip()) or FALLBACK
        return ans

    # Low confidence → refuse
    return FALLBACK

# Optional: local CLI for quick testing
if __name__ == "__main__":
    print("Type your question (Ctrl+C to quit).")
    try:
        while True:
            q = input("You: ").strip()
            if q.lower() == "exit":
                break
            print("Bot:", get_response(q))
    except KeyboardInterrupt:
        pass
