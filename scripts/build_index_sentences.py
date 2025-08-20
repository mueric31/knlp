import json, pickle, re
from pathlib import Path
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Split on sentence punctuation or newlines
SPLIT_RE = re.compile(r"(?<=[\.\?\!…])\s+|\n+")

def split_sentences(text: str):
    return [s.strip() for s in SPLIT_RE.split(text) if s and s.strip()]

def load_synonyms(path: Path | None) -> dict[str, list[str]]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for k, v in data.items():
        key = str(k).lower().strip()
        out[key] = [str(x).lower().strip() for x in (v or []) if str(x).strip()]
    return out

def augment_with_synonyms(sentence: str, syn: dict[str, list[str]]) -> str:
    """Append matched synonyms as extra terms to help TF-IDF match paraphrases."""
    s_lower = sentence.lower()
    extras = []
    for k, vs in syn.items():
        if k in s_lower:
            extras.extend(vs)
    if extras:
        return sentence + " || " + " ".join(sorted(set(extras)))
    return sentence

def main():
    ap = argparse.ArgumentParser(description="Build sentence-level TF-IDF index from sections.json")
    ap.add_argument("--sections", default="../data/sections.json", help="Path to sections JSON")
    ap.add_argument("--out", default="../data/tfidf_sent.pkl", help="Output pickle")
    ap.add_argument("--synonyms", default="../config/synonyms.json", help="Optional synonyms JSON")
    ap.add_argument("--ngmin", type=int, default=3, help="min char n-gram")
    ap.add_argument("--ngmax", type=int, default=5, help="max char n-gram")
    args = ap.parse_args()

    sections = json.loads(Path(args.sections).read_text(encoding="utf-8"))
    if not isinstance(sections, list) or not sections:
        raise ValueError("sections.json must be a non-empty list of {id,title,content}")

    synonyms = load_synonyms(Path(args.synonyms))

    items, docs = [], []
    for s in sections:
        title = (s.get("title") or "").strip()
        for sent in split_sentences(s.get("content", "")):
            items.append({"section_id": s.get("id","?"), "title": title, "sentence": sent})
            doc = f"{title} || {augment_with_synonyms(sent, synonyms)}"
            docs.append(doc)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(args.ngmin, args.ngmax), lowercase=True)
    X = vec.fit_transform(docs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"vectorizer": vec, "X": X, "items": items}, f)

    print(f"✅ Indexed {len(items)} sentences -> {out}")

if __name__ == "__main__":
    main()
