import fitz  # PyMuPDF
import re, json, sys
from pathlib import Path
import argparse

# Headings like: "1 ", "1.", "1) ", "1.1 ", "1.2.3 "
SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\s*[\.\)])?\s+(.*)")
PAGE_NO_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")   # e.g., "3/12"
LONE_NUM_RE = re.compile(r"^\s*\d+\s*$")            # e.g., "12"

def preprocess(raw_text: str) -> str:
    t = raw_text.replace("\u00AD", "")              # soft hyphen
    t = re.sub(r"[ \t]+", " ", t)                   # collapse spaces
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # de-hyphenate across linebreaks
    return t

def extract_sections(pdf_path: Path, skip_empty=True, min_content_chars=10):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    text = preprocess(text)
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
                if (not skip_empty) or (len(current["content"]) >= min_content_chars):
                    sections.append(current)
            sec_id, title = m.group(1), m.group(2).strip()
            current = {"id": sec_id, "title": title, "content": ""}
        else:
            if current is None:
                current = {"id": "0", "title": "Intro", "content": ""}
            current["content"] += line + "\n"

    if current:
        current["content"] = current["content"].strip()
        if (not skip_empty) or (len(current["content"]) >= min_content_chars):
            sections.append(current)

    return sections

def find_pdf(explicit_pdf: str | None, folder: Path, prefer_name: str | None, recursive: bool) -> Path:
    if explicit_pdf:
        p = Path(explicit_pdf)
        if not p.exists():
            sys.exit(f"❌ PDF not found: {p}")
        return p

    candidates = [folder, folder / "raw_text"]
    win_data = Path("C:/knlp/data")
    if win_data.exists() and win_data not in candidates:
        candidates.append(win_data)

    pdfs: list[Path] = []
    for d in candidates:
        if not d.exists(): 
            continue
        if recursive:
            pdfs.extend(d.rglob("*.pdf"))
        else:
            pdfs.extend(d.glob("*.pdf"))

    if not pdfs:
        sys.exit("❌ No PDFs found. Tried: " + ", ".join(str(c) for c in candidates))

    if prefer_name:
        for p in pdfs:
            if p.name.lower() == prefer_name.lower():
                print(f"📄 Using preferred PDF: {p}")
                return p

    print("📄 Found PDFs:\n  " + "\n  ".join(str(p) for p in pdfs))
    print(f"➡️  Using: {pdfs[0]}")
    return pdfs[0]

def main():
    ap = argparse.ArgumentParser(description="Split PDF into numbered sections (1, 1.1, 1.2.3…).")
    ap.add_argument("--pdf", help=r"Path to PDF (e.g., ..\data\imirire.pdf). If omitted, auto-detect.")
    ap.add_argument("--folder", default="../data", help="Folder to search when --pdf is omitted")
    ap.add_argument("--prefer", default="imirire.pdf", help="Preferred filename if multiple PDFs are found")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders for PDFs")
    ap.add_argument("--out", default="../data/sections.json", help="Output JSON with sections")
    ap.add_argument("--min-content", type=int, default=10, help="Min chars to keep a section")
    ap.add_argument("--keep-empty", action="store_true", help="Keep empty/short sections")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    pdf = find_pdf(args.pdf, Path(args.folder), args.prefer, args.recursive)
    secs = extract_sections(pdf, skip_empty=not args.keep_empty, min_content_chars=args.min_content)
    out.write_text(json.dumps(secs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Saved {len(secs)} sections from {pdf} -> {out}")

if __name__ == "__main__":
    main()
