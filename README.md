# Multimodal Agentic RAG

An AI assistant that answers questions grounded in your own **PDFs, images,
audio, video, and websites** — using only free, open technologies (no
OpenAI, no paid APIs). Retrieval is hybrid (vector + keyword), reranked with
a cross-encoder, and orchestrated by a **LangGraph** agent that streams
cited, confidence-scored answers into a **Streamlit** chat UI.

---

## Table of contents

- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Folder structure](#folder-structure)
- [Installation](#installation)
- [API keys & configuration](#api-keys--configuration)
- [Running locally](#running-locally)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Deploying with Docker](#deploying-with-docker)
- [Screenshots](#screenshots)
- [Future improvements](#future-improvements)

---

## Architecture

### Ingestion pipeline

```
Upload (PDF / Image / Audio / Video) or Website URL
        │
        ▼
 Planner: detect source type
        │
        ├── PDF     → PyMuPDF text extraction (+ Tesseract OCR fallback for scanned pages)
        ├── Image   → OpenCV pre-processing → Tesseract OCR
        ├── Audio   → OpenAI Whisper (local, open-source) transcription
        ├── Video   → FFmpeg audio extraction → Whisper, + OpenCV keyframe sampling → OCR
        └── Website → Requests + BeautifulSoup readable-text extraction
        │
        ▼
   Chunking (RecursiveCharacterTextSplitter, overlap-aware)
        │
        ▼
   Embedding (Nomic Embed, local via sentence-transformers)
        │
        ▼
   Storage: raw chunks → SQLite (also powers BM25) | vectors → Qdrant
```

### Query (agent) pipeline — LangGraph

```
START
  │
  ▼
rewrite_query        (Groq LLM resolves pronouns/context using chat history)
  │
  ▼
hybrid_retrieve       (Qdrant vector search + BM25 keyword search, score-fused)
  │
  ▼
rerank                (BAAI/bge-reranker-base cross-encoder)
  │
  ▼
generate_answer       (Groq llama-3.3-70b-versatile, streamed token-by-token)
  │
  ▼
build_citations       (numbered source list + heuristic confidence score)
  │
  ▼
END
```

Ingestion and querying are deliberately decoupled: ingestion is a
potentially slow, one-off batch job triggered from the sidebar's **Process**
button, while querying is a fast, low-latency LangGraph invocation per chat
turn. Both share the same planner/tool abstractions in
`backend/agent/tools.py`.

---

## Tech stack

| Concern            | Technology                                   | Cost |
|---------------------|-----------------------------------------------|------|
| Frontend            | Streamlit                                     | Free |
| Backend API         | FastAPI                                       | Free |
| Agent orchestration | LangGraph + LangChain                         | Free |
| LLM                 | Groq API — `llama-3.3-70b-versatile`          | Free tier |
| Embeddings          | Nomic Embed (`nomic-embed-text-v1.5`)         | Free, local |
| Vector DB           | Qdrant Cloud free tier (or local fallback)    | Free |
| Reranker            | `BAAI/bge-reranker-base`                      | Free, local |
| PDF parsing         | PyMuPDF                                        | Free |
| OCR                 | Tesseract OCR                                  | Free |
| Images              | Pillow + OpenCV                                | Free |
| Audio               | OpenAI Whisper (open-source, local)            | Free |
| Video               | OpenCV + FFmpeg                                | Free |
| Website loading     | Requests + BeautifulSoup                       | Free |
| Keyword search      | BM25 (`rank-bm25`)                             | Free |
| Metadata DB         | SQLite                                         | Free |

---

## Folder structure

```
multimodal_agentic_rag/
├── app.py                      # Streamlit UI (entry point)
├── backend/
│   ├── main.py                 # FastAPI app (optional HTTP interface)
│   ├── agent/
│   │   ├── graph.py            # LangGraph StateGraph wiring
│   │   ├── nodes.py            # Node functions (rewrite/retrieve/rerank/generate/cite)
│   │   ├── tools.py            # LangChain @tool wrappers (ingestion + retrieval)
│   │   ├── planner.py          # Source-type routing + query rewriting
│   │   └── memory.py           # SQLite-backed conversation memory
│   ├── retrieval/
│   │   ├── hybrid.py           # Vector + BM25 score fusion
│   │   ├── bm25.py             # BM25 keyword index
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   └── qdrant_store.py     # Qdrant client wrapper
│   ├── ingestion/
│   │   ├── pdf_loader.py       # PyMuPDF + OCR fallback
│   │   ├── image_loader.py     # OpenCV + Tesseract OCR
│   │   ├── audio_loader.py     # Whisper transcription
│   │   ├── video_loader.py     # FFmpeg + Whisper + keyframe OCR
│   │   ├── web_loader.py       # Requests + BeautifulSoup
│   │   ├── chunking.py         # Text splitting
│   │   └── pipeline.py         # End-to-end orchestration per source type
│   ├── models/
│   │   ├── groq_llm.py         # Groq chat-completions client (sync + streaming)
│   │   └── embeddings.py       # Nomic Embed wrapper
│   ├── database/
│   │   └── sqlite.py           # Documents, chunks, sessions, messages
│   └── utils/
│       ├── config.py           # Pydantic settings (env-driven)
│       ├── logger.py           # Rotating file + stdout logging
│       └── helpers.py          # Shared small utilities
├── data/                       # Runtime data (SQLite db, local Qdrant, uploads, logs)
├── .env.example
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Installation

### 1. Prerequisites

- Python 3.12
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on your `PATH`
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: install from the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`

### 2. Clone and set up a virtual environment

```bash
git clone <your-fork-url> multimodal_agentic_rag
cd multimodal_agentic_rag

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## API keys & configuration

Copy the example environment file and fill in your **free-tier** keys:

```bash
cp .env.example .env
```

| Variable          | Where to get it (free)                                    | Required? |
|--------------------|-------------------------------------------------------------|-----------|
| `GROQ_API_KEY`     | https://console.groq.com/keys                               | Yes       |
| `QDRANT_URL`       | https://cloud.qdrant.io (free 1GB cluster)                   | No — falls back to an embedded local vector store |
| `QDRANT_API_KEY`   | Same as above                                                | No        |

If `QDRANT_URL` is left blank, the app automatically uses a local, on-disk
Qdrant instance stored under `data/qdrant_local/` — no cloud account needed
to try the app out.

---

## Running locally

**Streamlit UI (recommended, all-in-one):**

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**FastAPI backend (optional, for programmatic access):**

```bash
python -m backend.main
# or: uvicorn backend.main:app --reload
```

The API docs are then available at http://localhost:8000/docs.

---

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and click **New app**.
3. Select your repository, branch, and set the main file path to `app.py`.
4. Under **Advanced settings → Secrets**, add your keys in TOML format:

   ```toml
   GROQ_API_KEY = "your-groq-key"
   QDRANT_URL = "https://your-cluster.qdrant.io"
   QDRANT_API_KEY = "your-qdrant-key"
   ```

5. Deploy. The first run will download the embedding/reranker/Whisper
   models from the Hugging Face Hub, which may take a few minutes.

> **Note:** Streamlit Community Cloud's default containers do not have
> Tesseract OCR or FFmpeg preinstalled. Add a `packages.txt` file (Streamlit
> Cloud reads this automatically) containing:
> ```
> tesseract-ocr
> ffmpeg
> ```

---

## Deploying with Docker

```bash
docker build -t multimodal-agentic-rag .
docker run -p 8501:8501 --env-file .env multimodal-agentic-rag
```

---

## Screenshots

_Add screenshots of the running app here, e.g.:_

- `docs/screenshots/chat-interface.png`
- `docs/screenshots/sidebar-ingestion.png`
- `docs/screenshots/citations-expander.png`

---

## Future improvements

- Async ingestion queue (Celery/RQ) so large videos don't block the UI thread.
- Multi-user auth and per-user document isolation.
- Support for additional loaders (DOCX, PPTX, CSV).
- Configurable reranker top-k and hybrid alpha from the UI.
- Automatic re-chunking strategy selection based on document structure
  (e.g. markdown-aware splitting for scraped docs).
- Evaluation harness (RAGAS) for retrieval/answer quality regression testing.
