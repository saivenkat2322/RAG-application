# Local RAG Website

A fully local Retrieval-Augmented Generation (RAG) web app:
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) — runs on CPU, no API key.
- **Vector store**: FAISS, in-memory (resets when the server restarts).
- **LLM**: [Ollama](https://ollama.com) running a local open-source model (e.g. `llama3.2`).
- **Frontend**: plain HTML/JS chat UI served by the same Flask app.

No cloud API keys are required — everything runs on your machine.

## 1. Install Ollama and pull a model

```backend
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull a small, fast model (swap for any model you like)
ollama pull llama3.2

# Start the Ollama server (keep this running in its own terminal)
ollama serve
```

## 2. Set up the backend

Unzip the project, then open a terminal inside that unzipped folder (it should contain
`app.py`, `rag.py`, `requirements.txt`, and a `frontend` folder):

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The first run downloads the embedding model (~80MB) automatically.
The server starts on **http://localhost:5000**.

## 3. Open the website

Visit **http://localhost:5000** in your browser — the Flask app also serves the frontend directly.
(If you prefer to run the frontend separately, e.g. via `python -m http.server` inside `frontend/`,
update `API_BASE` at the top of `index.html`'s `<script>` block to `http://localhost:5000`.)

## 4. Use it

1. Drag a `.txt`, `.md`, or `.pdf` file into the sidebar — it gets chunked, embedded, and added to the FAISS index.
2. Ask a question in the chat box. The app retrieves the most relevant chunks and asks the local LLM to answer using only that context.
3. Each answer shows which source chunks were used.

## Project structure

```
rag-clean/
├── app.py            # Flask API (upload, chat, health)
├── rag.py             # Embedding, FAISS index, retrieval, generation
├── requirements.txt
├── frontend/
│   └── index.html     # Chat UI (vanilla HTML/CSS/JS)
└── README.md
```

## Configuration

Environment variables (optional):
- `OLLAMA_URL` — default `http://localhost:11434/api/generate`
- `OLLAMA_MODEL` — default `llama3.2`

Tunable constants in `rag.py`:
- `CHUNK_SIZE`, `CHUNK_OVERLAP` — how documents are split
- `TOP_K` — how many chunks are retrieved per question

## Notes & next steps

- **Persistence**: the FAISS index is in-memory only. For a production setup, swap in
  `faiss.write_index`/`read_index` to persist to disk, or use a proper vector DB
  (Chroma, Qdrant, Weaviate, pgvector) for multi-user or larger-scale use.
- **Hosting publicly**: this is built to run locally. To put it on the internet, you'd typically
  deploy the Flask backend on a VM/container (EC2, Fly.io, Render, etc.) — note that Ollama needs
  real CPU/GPU resources, so serverless platforms aren't a great fit. The frontend (`index.html`)
  could alternatively be hosted as a static site (e.g. S3 + CloudFront) pointing at the backend's URL.
- **Swapping models**: any Ollama-supported model works — just change `OLLAMA_MODEL` and
  `ollama pull <model>`.
- **Security**: if exposing this beyond localhost, add authentication, file size limits, and
  rate limiting before going further.
