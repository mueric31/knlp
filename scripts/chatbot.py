#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, pickle, re
from pathlib import Path
import numpy as np
import torch, tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from gpt_model import GPTConfig, GPTLanguageModel

FALLBACK = "nta makuru ahagije nari nagira"

INSTRUCTIONS = (
    "Subiza mu nteruro nkeya GUKORESHA gusa amakuru ari muri CONTEXT.\n"
    "NIBA CONTEXT idafite igisubizo, andika: 'nta makuru ahagije nari nagira'."
)

SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!…])\s+|\n+")

def first_sentence(txt: str) -> str:
    return re.split(r"(?<=[\.\?\!])\s", txt.strip(), maxsplit=1)[0].strip()

def split_sentences(text: str):
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s and s.strip()]

def load_model(ckpt, cfg_path, device):
    cfg = GPTConfig(**json.loads(Path(cfg_path).read_text(encoding="utf-8")))
    model = GPTLanguageModel(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, cfg

def load_index(path: Path):
    data = pickle.load(open(path, "rb"))
    for k in ("vectorizer","X","items"):
        if k not in data:
            raise ValueError("Bad index file; rebuild with build_index_sentences.py")
    return data

def load_sections_map(path: Path):
    if not path.exists(): return {}
    arr = json.loads(path.read_text(encoding="utf-8"))
    return {str(s.get("id","")).strip(): {"title": s.get("title","").strip(),
                                          "content": s.get("content","").strip()}
            for s in arr}

def load_synonyms(path: Path|None) -> dict[str, list[str]]:
    if not path or not path.exists(): return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        key = str(k).lower().strip()
        out[key] = [str(x).lower().strip() for x in (v or []) if str(x).strip()]
    return out

ALLOW_SINGLE_WORD = {"ikawa", "inzoga", "itabi", "ibiyobyabwenge"}  # useful 1-word anchors

def expand_query(q: str, synonyms: dict[str, list[str]]):
    """
    If user's question mentions any synonym key (or its variants), append those
    variants into the query string to help the vectorizer. Returns (expanded_q, matched_keys).
    We avoid extremely broad keys by requiring multi-word keys OR whitelisted single words.
    """
    q_low = q.lower()
    matched = []
    addon_terms = []
    for key, vars_ in synonyms.items():
        is_phrase = (" " in key) or ("’" in key) or ("'" in key)
        is_allowed_single = key in ALLOW_SINGLE_WORD or len(key) >= 6
        if (is_phrase or is_allowed_single):
            # if key or any variant is present in query -> expand
            if key in q_low or any(v in q_low for v in vars_):
                matched.append(key)
                addon_terms.extend([key] + vars_)
    if addon_terms:
        expanded = q + " || " + " ".join(sorted(set(addon_terms)))
    else:
        expanded = q
    return expanded, matched

def retrieve(expanded_q, idx, matched_keys, synonyms, title_boost=1.25, topk=3):
    vec, X, items = idx["vectorizer"], idx["X"], idx["items"]
    sims = cosine_similarity(vec.transform([expanded_q]), X)[0]

    # Light title boost: if a matched concept also appears in the item's title, upweight it.
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

    order = np.argsort(sims)[::-1][:topk]
    picks = [{"score": float(sims[i]), **items[i]} for i in order]
    return picks

def build_context(picks):
    lines = []
    for p in picks:
        sec = p.get("section_id","?")
        ttl = (p.get("title") or "").strip()
        sent = (p.get("sentence") or "").strip()
        lines.append(f"- [{sec}] {ttl}: {sent}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Hybrid extractive + strict RAG inference (synonym-expanded queries).")
    ap.add_argument("--ckpt", default="../model/checkpoint/model.pt")
    ap.add_argument("--cfg",  default="../model/checkpoint/config.json")
    ap.add_argument("--index", default="../data/tfidf_sent.pkl")
    ap.add_argument("--sections", default="../data/sections.json", help="sections.json to show full section content")
    ap.add_argument("--synonyms", default="../config/synonyms.json", help="synonyms for query expansion")
    ap.add_argument("--hi", type=float, default=0.40, help="high-confidence threshold (extractive)")
    ap.add_argument("--mid", type=float, default=0.28, help="mid-confidence threshold (RAG gen)")
    ap.add_argument("--context_topk", type=int, default=2)
    ap.add_argument("--title_boost", type=float, default=1.25)
    ap.add_argument("--high_mode", choices=["sentence","section","section_sentences"], default="section_sentences")
    ap.add_argument("--n_sentences", type=int, default=0, help="If section_sentences, 0=all")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repeat_penalty", type=float, default=1.03)
    ap.add_argument("--freq_penalty", type=float, default=0.3)
    ap.add_argument("--show_source", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    model, cfg = load_model(args.ckpt, args.cfg, device)
    idx = load_index(Path(args.index))
    sections_map = load_sections_map(Path(args.sections))
    synonyms = load_synonyms(Path(args.synonyms))

    print(f"🔎 Using sentence index: {args.index}")
    print("Type your question (Ctrl+C to quit).")
    while True:
        q = input("You: ").strip()
        if not q:
            print(FALLBACK); continue

        expanded_q, matched_keys = expand_query(q, synonyms)
        picks = retrieve(expanded_q, idx, matched_keys, synonyms,
                         title_boost=args.title_boost, topk=max(1, args.context_topk))
        if not picks:
            print(FALLBACK); continue

        best = picks[0]
        score = best["score"]
        sec = best.get("section_id","?")
        ttl = (best.get("title") or "").strip()
        sent = (best.get("sentence") or "").strip()

        if args.show_source:
            print(f"📌 score={score:.3f}  section={sec}  title={ttl}")

        # 1) High confidence → extractive
        if score >= args.hi:
            if args.high_mode == "sentence":
                print("Bot:", sent)
            elif args.high_mode == "section":
                sec_data = sections_map.get(str(sec))
                if sec_data and sec_data.get("content"):
                    print("Bot:", f"{sec_data['title']}\n{sec_data['content']}")
                else:
                    print("Bot:", sent)
            else:  # section_sentences
                sec_data = sections_map.get(str(sec))
                if sec_data and sec_data.get("content"):
                    sents = split_sentences(sec_data["content"])
                    if args.n_sentences > 0:
                        sents = sents[:args.n_sentences]
                    print("Bot:", " ".join(sents))
                else:
                    print("Bot:", sent)
            continue

        # 2) Medium confidence → strict RAG (short, from CONTEXT only)
        if score >= args.mid:
            ctx = build_context(picks)
            prompt = f"{INSTRUCTIONS}\n\nQ: {q}\nCONTEXT:\n{ctx}\nA:"
            ids = enc.encode(prompt) or enc.encode(" ")
            x = torch.tensor([ids], dtype=torch.long, device=device)
            if x.size(1) > cfg.block_size: x = x[:, -cfg.block_size:]

            with torch.no_grad():
                out = model.generate(
                    x,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repeat_penalty=args.repeat_penalty,
                    freq_penalty=args.freq_penalty,
                )
            text = enc.decode(out[0].tolist())
            ans = text.split("A:", 1)[-1].strip()
            print("Bot:", first_sentence(ans) or FALLBACK)
            continue

        # 3) Low confidence → refuse
        print(FALLBACK)

if __name__ == "__main__":
    main()
