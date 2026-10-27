"""
rag.py - Core RAG (Retrieval-Augmented Generation) engine.

Uses:
- sentence-transformers (all-MiniLM-L6-v2) for embeddings (local, free, no API key)
- FAISS (in-memory) for vector similarity search
- Ollama (local LLM server) for text generation

Prereqs:
- pip install -r requirements.txt
- Install Ollama (https://ollama.com) and pull a model, e.g.:
    ollama pull llama3.2
"""

import os
import re
import uuid
import requests
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 80     # overlap between chunks
TOP_K = 4              # number of chunks to retrieve per query


class RAGEngine:
    def __init__(self):
        print("Loading embedding model (first run downloads it)...")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)  # cosine similarity via normalized vectors
        self.chunks = []        # list of text chunks
        self.metadata = []      # list of dicts: {source, chunk_id}
        print("RAG engine ready.")

    # ---------- Document ingestion ----------

    def _chunk_text(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP
        return [c for c in chunks if c.strip()]

    def _extract_text(self, filepath):
        if filepath.lower().endswith(".pdf"):
            reader = PdfReader(filepath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def add_document(self, filepath, source_name=None):
        source_name = source_name or os.path.basename(filepath)
        text = self._extract_text(filepath)
        new_chunks = self._chunk_text(text)
        if not new_chunks:
            return 0

        embeddings = self.embedder.encode(new_chunks, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")
        self.index.add(embeddings)

        for i, chunk in enumerate(new_chunks):
            self.chunks.append(chunk)
            self.metadata.append({"source": source_name, "chunk_id": i, "id": str(uuid.uuid4())})

        return len(new_chunks)

    def doc_count(self):
        return len(set(m["source"] for m in self.metadata))

    def chunk_count(self):
        return len(self.chunks)

    # ---------- Retrieval ----------

    def retrieve(self, query, top_k=TOP_K):
        if self.index.ntotal == 0:
            return []
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "score": float(score),
            })
        return results

    # ---------- Generation ----------

    def _build_prompt(self, query, contexts):
        context_block = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in contexts
        )
        prompt = f"""You are a helpful assistant answering questions using ONLY the context below.
If the answer isn't in the context, say you don't know based on the provided documents.

Context:
{context_block}

Question: {query}

Answer:"""
        return prompt

    def generate_answer(self, query, top_k=TOP_K):
        contexts = self.retrieve(query, top_k=top_k)
        if not contexts:
            return {
                "answer": "No documents have been indexed yet. Please upload a document first.",
                "sources": [],
            }

        prompt = self._build_prompt(query, contexts)

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            answer = (
                "[LLM not reachable] Could not connect to Ollama at "
                f"{OLLAMA_URL}. Make sure Ollama is running (`ollama serve`) "
                f"and the model is pulled (`ollama pull {OLLAMA_MODEL}`).\n\n"
                "Here are the most relevant passages found instead:\n\n"
                + "\n---\n".join(c["text"] for c in contexts)
            )
        except Exception as e:
            answer = f"[Error generating answer] {e}"

        return {
            "answer": answer,
            "sources": [{"source": c["source"], "score": round(c["score"], 3)} for c in contexts],
        }
