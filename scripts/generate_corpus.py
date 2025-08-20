import argparse, json, re, sys, pickle
from pathlib import Path

# --- optional deps for indexes ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity  # not used here, just to confirm import
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# --- optional dep for PDF extraction ---
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

# ---------- helpers ----------
SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\s*[\.\)])?\s+(.*)")
PAGE_NO_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")   # e.g., "3/12"
LONE_NUM_RE = re.compile(r"^\s*\d+\s*$")            # e.g., "12"
SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!…])\s+|\n+")

DEFAULT_SYNONYMS = {
    "inzoga": ["ibisindisha", "alcohol"],
    "ibisindisha": ["inzoga"],
    "ikawa": ["coffee"],
    "itabi": ["sigara", "tobacco"],
    "sigara": ["itabi"],
    "ibiyobyabwenge": ["drogue", "drugs"],
    "indyo ikwiye": ["indyo yuzuye", "ibyo kurya byiza", "ibiribwa byiza"],
    "indyo yuzuye": ["indyo ikwiye"],
    "inshuro": ["umubare w'inshuro", "kangahe"],
    "kwirinda": ["ntibyemewe", "birabujijwe", "bibujijwe"],
    "ibyo kurya": ["ibiribwa"],
    "inyunganiramirire": ["supplements", "vitamini"]
}

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def preprocess_pdf_text(raw_text: str) -> str:
    t = raw_text.replace("\u00AD", "")        # soft hyphen
    t = re.sub(r"[ \t]+", " ", t)            # collapse spaces
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)   # de-hyphenate across linebreaks
    return t

def extract_pdf_text(pdf_path: Path) -> str:
    if not HAVE_PYMUPDF:
        sys.exit("❌ PyMuPDF not installed. Run: pip install pymupdf")
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return preprocess_pdf_text(text)

def split_sections_from_text(text: str):
    lines = text.splitlines()
    sections, current = [], None
    for raw in lines:
        line = raw.strip()
        if not line: 
            continue
        if PAGE_NO_RE.match(line) or LONE_NUM_RE.match(line):
            continue
        m = SECTION_RE.match(line)
        if m:
            if current:
                current["content"] = current["content"].strip()
                if current["content"]:
                    sections.append(current)
            sec_id, title = m.group(1), m.group(2).strip()
            current = {"id": sec_id, "title": title, "content": ""}
        else:
            if current is None:
                current = {"id": "0", "title": "Intro", "content": ""}
            current["content"] += line + "\n"
    if current:
        current["content"] = current["content"].strip()
        if current["content"]:
            sections.append(current)
    return sections

def write_sections_json(sections, out_path: Path):
    out_path.write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")

def make_clean_txt(sections) -> str:
    # Simple, ordered, readable training text
    out_lines = []
    # Optional top title:
    out_lines.append("Imirire myiza ku mugore utwite")
    out_lines.append("")

    for s in sections:
        title = (s.get("title") or "").strip()
        content = (s.get("content") or "").strip()
        if not title and not content: 
            continue
        if s.get("id") and s["id"] != "0":
            out_lines.append(f"{s['id']} {title}")
        else:
            out_lines.append(title if title else "Intro")
        # Normalize bullets visually (just reflow text)
        content = re.sub(r"\s{2,}", " ", content)
        out_lines.append(content)
        out_lines.append("")
    return "\n".join(out_lines).strip() + "\n"

def split_sentences(text: str):
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s and s.strip()]

def load_or_create_synonyms(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # normalize
            norm = {}
            for k, v in data.items():
                norm[str(k).lower().strip()] = [str(x).lower().strip() for x in (v or []) if str(x).strip()]
            return norm
        except Exception:
            pass
    # create default
    ensure_dir(path.parent)
    path.write_text(json.dumps(DEFAULT_SYNONYMS, ensure_ascii=False, indent=2), encoding="utf-8")
    return DEFAULT_SYNONYMS

def augment_with_synonyms(sentence: str, syn: dict) -> str:
    s_lower = sentence.lower()
    extras = []
    for k, vs in syn.items():
        if k in s_lower:
            extras.extend(vs)
    if extras:
        return sentence + " || " + " ".join(sorted(set(extras)))
    return sentence

def build_section_index(sections, out_path: Path):
    if not HAVE_SK:
        print("⚠️  scikit-learn not installed; skipping section index. Run: pip install scikit-learn")
        return
    docs = [f"{(s.get('title') or '').strip()}\n{(s.get('content') or '').strip()}" for s in sections]
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    X = vec.fit_transform(docs)
    with open(out_path, "wb") as f:
        pickle.dump({"vectorizer": vec, "X": X, "sections": sections}, f)
    print(f"✅ Built section index -> {out_path}")

def build_sentence_index(sections, syn_path: Path, out_path: Path, ngmin=3, ngmax=5):
    if not HAVE_SK:
        print("⚠️  scikit-learn not installed; skipping sentence index. Run: pip install scikit-learn")
        return
    synonyms = load_or_create_synonyms(syn_path)
    items, docs = [], []
    for s in sections:
        title = (s.get("title") or "").strip()
        for sent in split_sentences(s.get("content", "")):
            items.append({"section_id": s.get("id","?"), "title": title, "sentence": sent})
            docs.append(f"{title} || {augment_with_synonyms(sent, synonyms)}")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(ngmin, ngmax), lowercase=True)
    X = vec.fit_transform(docs)
    with open(out_path, "wb") as f:
        pickle.dump({"vectorizer": vec, "X": X, "items": items}, f)
    print(f"✅ Built sentence index -> {out_path}")

def qa_templates_for(title: str, content: str):
    """Return a list of (instruction, output) minimal Q/A pairs based on known titles; else generic."""
    t = title.lower()
    qa = []
    if "indyo ikwiye" in t or "indyo yuzuye" in t:
        qa.append(("Indyo ikwiye ku mugore utwite ni iyihe?", content))
        qa.append(("Ni ibihe byiciro by’ingenzi by’indyo ku mugore utwite?", content))
    elif "inshuro" in t:
        qa.append(("Inshuro umugore utwite agomba kurya ni zihe?", content))
        qa.append(("Kuki umugore utwite akwiriye kongera inshuro zo kurya?", content))
    elif "ibiribwa" in t or "ibinyobwa" in t or "bibujijwe" in t or "kwirinda" in t:
        qa.append(("Ibiribwa bibujijwe ku mugore utwite ni ibihe?", content))
        qa.append(("Ese umugore utwite yanywa ikawa?", content))
        qa.append(("Ese inzoga zemererwa ku mugore utwite?", content))
    elif "imiti" in t:
        qa.append(("Imiti ibujijwe ku mugore utwite ni iyihe?", content))
        qa.append(("Ese nshobora gufata umuti ntagiye kwa muganga?", content))
    elif "imikurire y’umwana" in t or "imikurire y'umwana" in t:
        qa.append(("Kumenya imikurire y’umwana bisobanura iki?", content))
        qa.append(("Ni iki gishobora gufasha gukumira ibibazo by’imikurire y’umwana?", content))
    else:
        qa.append((f"Sobanura: {title}", content))
    return qa

def write_qa_jsonl(sections, out_path: Path):
    lines = []
    for s in sections:
        title = (s.get("title") or "").strip()
        content = (s.get("content") or "").strip()
        for q, a in qa_templates_for(title, content):
            obj = {"instruction": q, "output": a}
            lines.append(json.dumps(obj, ensure_ascii=False))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ Wrote QA dataset -> {out_path}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate clean corpus, sections, QA, and indexes from a PDF.")
    ap.add_argument("--pdf", default=r"..\data\imirire.pdf", help=r'Path to your PDF (default: ..\data\imirire.pdf)')
    ap.add_argument("--out_dir", default="../data", help="Output folder for generated files")
    ap.add_argument("--make_qa", action="store_true", help="Also generate qa.jsonl")
    ap.add_argument("--build_indexes", action="store_true", help="Also build TF-IDF section & sentence indexes")
    ap.add_argument("--synonyms", default="../config/synonyms.json", help="Synonyms JSON (auto-created if missing)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    syn_path = Path(args.synonyms)

    if not pdf_path.exists():
        sys.exit(f"❌ PDF not found: {pdf_path}")
    ensure_dir(out_dir)
    ensure_dir(syn_path.parent)

    # 1) extract + split
    print("📄 Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    print("✂️  Splitting into sections...")
    sections = split_sections_from_text(text)
    if not sections:
        sys.exit("❌ No sections detected. Ensure headings like '1 ', '1.1 ' are present.")

    # 2) write sections.json
    sections_json = out_dir / "sections.json"
    write_sections_json(sections, sections_json)
    print(f"✅ Saved sections -> {sections_json}")

    # 3) clean training text
    clean_txt_path = out_dir / "kinyarwanda_clean.txt"
    clean_txt = make_clean_txt(sections)
    clean_txt_path.write_text(clean_txt, encoding="utf-8")
    print(f"✅ Saved clean text -> {clean_txt_path}")

    # 4) optional QA
    if args.make_qa:
        qa_path = out_dir / "qa.jsonl"
        write_qa_jsonl(sections, qa_path)

    # 5) optional indexes
    if args.build_indexes:
        sec_idx = out_dir / "tfidf.pkl"
        sent_idx = out_dir / "tfidf_sent.pkl"
        build_section_index(sections, sec_idx)
        build_sentence_index(sections, syn_path, sent_idx)

    print("🎉 Done.")

if __name__ == "__main__":
    main()
