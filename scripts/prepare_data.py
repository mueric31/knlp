from pathlib import Path
import argparse
import numpy as np
import tiktoken

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../data/kinyarwanda.txt")
    ap.add_argument("--out", dest="out", default="../data/kinyarwanda.bin")
    ap.add_argument("--vocab", default="gpt2", help="tiktoken vocab name (default: gpt2)")
    args = ap.parse_args()

    text = Path(args.inp).read_text(encoding="utf-8")
    enc = tiktoken.get_encoding(args.vocab)
    tokens = enc.encode(text)
    np.array(tokens, dtype=np.uint16).tofile(args.out)
    print(f"✅ Saved {len(tokens)} tokens to {args.out}")

if __name__ == "__main__":
    main()
