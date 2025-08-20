#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer_extractive.py
-------------------
Pure extractive Q&A using your sentence-level TF-IDF index.
It returns the most relevant sentence(s) from your corpus — no generation.

Usage (from C:\knlp\scripts):
  python infer_extractive.py --index ..\data\tfidf_sent.pkl --threshold 0.30 --topk 1 --show_meta
"""

import argparse
import pickle
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Extractive Q&A from a sentence-level TF-IDF index.")
    ap.add_argument("--index", default="../data/tfidf_sent.pkl",
                    help="Path to sentence TF-IDF index built by build_index_sentences.py")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="Minimum cosine similarity to answer")
    ap.add_argument("--topk", type=int, default=1,
                    help="Return top-k sentences (>=1)")
    ap.add_argument("--show_meta", action="store_true",
                    help="Show section id/title and similarity scores")
    args = ap.parse_args()

    idx_path = Path(args.index)
    if not idx_path.exists():
        print(f"❌ Index not found: {idx_path}")
        print("   Build it first, e.g.:")
        print("   python build_index_sentences.py --sections ..\\data\\sections.json --out ..\\data\\tfidf_sent.pkl")
        return

    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        print("❌ scikit-learn not installed. Install with: pip install scikit-learn")
        return

    data = pickle.load(open(idx_path, "rb"))
    for key in ("vectorizer", "X", "items"):
        if key not in data:
            print("❌ Bad index file. Rebuild with build_index_sentences.py")
            return

    vec, X, items = data["vectorizer"], data["X"], data["items"]

    print(f"🔎 Using sentence index: {args.index}")
    print("Type your question (Ctrl+C to quit).")
    try:
        while True:
            q = input("You: ").strip()
            if not q:
                print("nta makuru ahagije nari nagira")
                continue

            sims = cosine_similarity(vec.transform([q]), X)[0]
            order = sims.argsort()[::-1]  # descending
            # filter by threshold
            kept = [i for i in order if sims[i] >= args.threshold]

            if not kept:
                print("nta makuru ahagije nari nagira")
                continue

            topk = max(1, args.topk)
            picks = kept[:topk]

            if len(picks) == 1:
                j = picks[0]
                it = items[j]
                sent = (it.get("sentence") or "").strip()
                if args.show_meta:
                    sec = it.get("section_id", "?")
                    ttl = (it.get("title") or "").strip()
                    print(f"📌 score={sims[j]:.3f}  section={sec}  title={ttl}")
                print("Bot:", sent)
            else:
                # show a short ranked list
                print("Bot:")
                for rank, j in enumerate(picks, 1):
                    it = items[j]
                    sent = (it.get("sentence") or "").strip()
                    if args.show_meta:
                        sec = it.get("section_id", "?")
                        ttl = (it.get("title") or "").strip()
                        print(f"  {rank}. ({sims[j]:.3f}) [{sec}] {ttl}: {sent}")
                    else:
                        print(f"  {rank}. {sent}")

    except KeyboardInterrupt:
        print("\n👋 Bye!")

if __name__ == "__main__":
    main()
