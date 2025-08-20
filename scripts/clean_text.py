from pathlib import Path
import argparse, re, json

def clean_text_str(
    text: str,
    titles_regex: str | None = None,
    use_auto_headings: bool = False,
    min_title_words: int = 2,
    max_title_words: int = 12,
) -> str:
    # 1) basic cleaning
    text = re.sub(r"\b\d+(\.\d+)*\b", " ", text)   # numbers like 1.2 or 1.2.3
    text = re.sub(r"[•\-–]+", " ", text)           # bullets/dashes
    text = re.sub(r"\s{2,}", " ", text)            # multi-spaces

    # 2) normalize section titles
    if titles_regex:
        text = re.sub(fr"({titles_regex})", r"\n\1\n", text, flags=re.I)
    elif use_auto_headings:
        # heuristic: short-ish lines that look like headings
        lines = text.split("\n")
        out = []
        for i, line in enumerate(lines):
            s = line.strip()
            words = s.split()
            is_heading = (
                min_title_words <= len(words) <= max_title_words
                and len(s) <= 80
                and not s.endswith(".")
                and (s[:1].isupper() or s[:1].isdigit())
            )
            if is_heading and s:
                if len(out) > 0 and out[-1].strip() != "":
                    out.append("")  # blank line before
                out.append(s)
                out.append("")      # blank line after
            else:
                out.append(s)
        text = "\n".join(out)

    return (text.strip() + "\n")

def load_titles_from_file(path: Path) -> list[str]:
    # supports .txt (one regex per line) or .json (array of regex strings)
    if path.suffix.lower() == ".json":
        arr = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(arr, list):
            raise ValueError("JSON titles file must be a list of regex strings.")
        return [str(x) for x in arr]
    else:
        # txt: non-empty, non-comment lines
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
        return lines

def main():
    ap = argparse.ArgumentParser(description="Clean Kinyarwanda text without hardcoded titles.")
    ap.add_argument("--in", dest="inp", default="../data/kinyarwanda.txt", help="Input TXT")
    ap.add_argument("--out", dest="out", default=None, help="Output TXT (default: overwrite input)")
    ap.add_argument("--all", action="store_true", help="Clean all .txt in ../data (in place)")
    ap.add_argument("--titles", default=None,
                    help="Regex alternation for titles, e.g. '^Indyo.*|^Inshuro .*'")
    ap.add_argument("--titles-file", dest="titles_file", default=None,
                    help="Path to titles file (.txt lines or .json array)")
    ap.add_argument("--auto-headings", action="store_true",
                    help="Use heuristics to detect headings instead of explicit titles")
    ap.add_argument("--min-title-words", type=int, default=2)
    ap.add_argument("--max-title-words", type=int, default=12)
    ap.add_argument("--no-titles", action="store_true",
                    help="Skip title normalization entirely")
    args = ap.parse_args()

    # decide titles source
    titles_regex = None
    if not args.no_titles and not args.auto_headings:
        patterns = []
        if args.titles:
            patterns.append(args.titles)  # already alternation string
        if args.titles_file:
            patterns.extend(load_titles_from_file(Path(args.titles_file)))
        if patterns:
            # if any pattern lines provided, join them into one alternation
            titles_regex = "|".join(patterns)

    def clean_one(p: Path, out_path: Path | None):
        text = p.read_text(encoding="utf-8")
        cleaned = clean_text_str(
            text,
            titles_regex=titles_regex,
            use_auto_headings=args.auto_headings and not titles_regex,
            min_title_words=args.min_title_words,
            max_title_words=args.max_title_words,
        )
        dst = out_path or p
        dst.write_text(cleaned, encoding="utf-8")
        print(f"✅ Cleaned {p} -> {dst}")

    if args.all:
        for p in sorted(Path("../data").glob("*.txt")):
            if p.name.endswith(".cleaned.txt"): 
                continue
            clean_one(p, None)
    else:
        src = Path(args.inp)
        dst = Path(args.out) if args.out else None
        clean_one(src, dst)

if __name__ == "__main__":
    main()
