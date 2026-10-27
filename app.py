"""
app.py - Flask backend for the RAG website.

Endpoints:
  GET  /api/health        -> health check + index stats
  POST /api/upload        -> upload a .txt or .pdf document, indexes it
  POST /api/chat          -> { "query": "..." } -> { "answer": "...", "sources": [...] }

Run:
  pip install -r requirements.txt
  python app.py
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from rag import RAGEngine

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
ALLOWED_EXT = {"txt", "pdf", "md"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

engine = RAGEngine()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "documents_indexed": engine.doc_count(),
        "chunks_indexed": engine.chunk_count(),
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .txt, .md, and .pdf files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    try:
        n_chunks = engine.add_document(filepath, source_name=filename)
    except Exception as e:
        return jsonify({"error": f"Failed to process document: {e}"}), 500

    return jsonify({
        "message": f"Indexed '{filename}' successfully",
        "chunks_added": n_chunks,
        "total_chunks": engine.chunk_count(),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = (data or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is empty"}), 400

    result = engine.generate_answer(query)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
