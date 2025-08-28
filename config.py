from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def pick_path(primary: Path, fallback: Path) -> str:
    return str(primary if primary.exists() else fallback)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")

PDF_PATH   = os.getenv("PDF_PATH")   or pick_path(DATA/"imirire.pdf", DATA/"imirire.pdf")
FAISS_PATH = os.getenv("FAISS_PATH") or pick_path(DATA/"index.faiss", DATA/"index.faiss")
META_PATH  = os.getenv("META_PATH")  or pick_path(DATA/"meta.jsonl", DATA/"meta.jsonl")
SYN_PATH   = os.getenv("SYN_PATH")   or pick_path(DATA/"synonyms.json", DATA/"synonyms.json")

CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE", "900"))
OVERLAP     = int(os.getenv("OVERLAP",    "200"))
TOP_K       = int(os.getenv("TOP_K",      "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.15"))
