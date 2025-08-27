# Kinyarwanda RAG Chatbot (PDF-based)

This is a simple Retrieval-Augmented Generation (RAG) chatbot that answers **only** from your PDF.
It uses OpenAI embeddings + FAISS for retrieval, and `gpt-4o-mini` (configurable) to answer in Kinyarwanda.
If no relevant answer is found, it replies with: **"ntamakuru ndagira kuri iyi ngingo"**.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.sample .env
# then edit .env and set your OPENAI_API_KEY
```

## 2) Build the FAISS index from your PDF

The script reads `data/imirire.pdf`, chunks it, embeds, and saves `data/index.faiss` and `data/meta.jsonl`.

```bash
python build_index.py
```

## 3) Ask questions (in Kinyarwanda)

```bash
python chat.py "Ni izihe nkingo umugore utwite agomba gufata?"
python chat.py "Ibimenyetso bigaragaza ko utwite ni ibihe?"
```

You can keep asking in an interactive loop:

```bash
python chat.py
```

## 4) Configuration

- Edit `.env` to change models or chunking:
  - `EMBED_MODEL=text-embedding-3-small`
  - `CHAT_MODEL=gpt-4o-mini`
  - `CHUNK_SIZE=900`
  - `OVERLAP=200`
  - `TOP_K=5`
  - `SCORE_THRESHOLD=0.15`
- The synonyms file is loaded from `data/synonyms.json`. The query is **expanded** with related words
  to improve recall, then searched in FAISS.

## Notes
- Answers are **restricted to the PDF context**. If nothing relevant is found, the bot returns
  "ntamakuru ndagira kuri iyi ngingo".
- Tested with Python 3.10+. On Windows, `faiss-cpu` installs via pip on recent Python versions.
- If your PDF is large, first run `build_index.py` (it may take some minutes depending on size & network).
