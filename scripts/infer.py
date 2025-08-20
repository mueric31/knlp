import argparse, json, pickle, re
from pathlib import Path
import torch, tiktoken
from sklearn.metrics.pairwise import cosine_similarity

from gpt_model import GPTConfig, GPTLanguageModel

FALLBACK = "nta makuru ahagije nari nagira"

def load_model(ckpt, cfg_path, device):
    cfg = GPTConfig(**json.loads(Path(cfg_path).read_text(encoding="utf-8")))
    model = GPTLanguageModel(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, cfg

def load_sentence_index(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # expects {"vectorizer": vec, "X": X, "items": items}
    if not all(k in data for k in ("vectorizer","X","items")):
        raise ValueError("Index missing keys; build with build_index_sentences.py")
    return data

def retrieve(q, idx, topk=2):
    vec, X, items = idx["vectorizer"], idx["X"], idx["items"]
    qv = vec.transform([q])
    sims = cosine_similarity(qv, X)[0]
    order = sims.argsort()[::-1][:topk]
    picks = [{"score": float(sims[i]), **items[i]} for i in order]
    return picks

def build_context(picks):
    # Compact bullet list of top sentences, keep section id/title to guide the model
    lines = []
    for p in picks:
        sec = p.get("section_id","?")
        ttl = (p.get("title") or "").strip()
        sent = (p.get("sentence") or "").strip()
        if ttl: lines.append(f"[{sec}] {ttl}: {sent}")
        else:   lines.append(f"[{sec}] {sent}")
    return "\n".join(lines)

def first_sentence(txt: str) -> str:
    return re.split(r"(?<=[\.\?\!])\s", txt.strip(), maxsplit=1)[0].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../model/checkpoint/model.pt")
    ap.add_argument("--cfg",  default="../model/checkpoint/config.json")
    ap.add_argument("--index", default="../data/tfidf_sent.pkl",
                    help="Use the SENTENCE index built by build_index_sentences.py")
    ap.add_argument("--threshold", type=float, default=0.28,
                    help="min cosine similarity to answer")
    ap.add_argument("--context_topk", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repeat_penalty", type=float, default=1.05)
    ap.add_argument("--freq_penalty", type=float, default=0.25)
    ap.add_argument("--show_source", action="store_true")
    args = ap.parse_args()

    # load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    model, cfg = load_model(args.ckpt, args.cfg, device)
    idx = load_sentence_index(Path(args.index))

    print(f"🔎 Using sentence index: {args.index}")
    print("Type your question (Ctrl+C to quit).")
    while True:
        q = input("You: ").strip()
        if not q:
            print(FALLBACK); continue

        picks = retrieve(q, idx, topk=max(1, args.context_topk))
        best = picks[0] if picks else None
        if not best or best["score"] < args.threshold:
            print(FALLBACK); continue

        ctx = build_context(picks)
        if args.show_source:
            print(f"📌 source score={best['score']:.2f}  section={best.get('section_id','?')}  title={best.get('title','')}")

        prompt = f"Q: {q}\nCONTEXT:\n{ctx}\nA:"
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
        ans = first_sentence(ans)  # keep it tight
        print("Bot:", ans)

if __name__ == "__main__":
    main()
