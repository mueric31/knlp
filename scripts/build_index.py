import json
import pickle
from pathlib import Path
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    ap = argparse.ArgumentParser(description="Build a TF-IDF index from sections.json")
    ap.add_argument("--sections", default="../data/sections.json", help="Path to sections JSON")
    ap.add_argument("--out", default="../data/tfidf.pkl", help="Output pickle path")
    ap.add_argument("--ngram_min", type=int, default=1, help="Min n-gram")
    ap.add_argument("--ngram_max", type=int, default=2, help="Max n-gram")
    args = ap.parse_args()

    sections_path = Path(args.sections)
    if not sections_path.exists():
        raise FileNotFoundError(f"Sections file not found: {sections_path}")

    sections = json.loads(sections_path.read_text(encoding="utf-8"))
    if not isinstance(sections, list) or not sections:
        raise ValueError("sections.json must be a non-empty JSON array of {id,title,content}")

    docs = []
    for s in sections:
        title = (s.get("title") or "").strip()
        content = (s.get("content") or "").strip()
        docs.append(f"{title}\n{content}")

    vectorizer = TfidfVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        lowercase=True,
    )
    X = vectorizer.fit_transform(docs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "X": X, "sections": sections}, f)

    print(f"✅ Indexed {len(docs)} sections -> {out_path}")

if __name__ == "__main__":
    main()
