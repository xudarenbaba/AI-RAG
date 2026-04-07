from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from rag_assistant import RagAssistant


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    assistant = RagAssistant("config.yaml")

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/health")
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok"}), 200

    @app.post("/ingest/text")
    def ingest_text() -> tuple[Any, int]:
        body = request.get_json(force=True, silent=True) or {}
        text = str(body.get("text", "")).strip()
        source = str(body.get("source", "manual_input")).strip() or "manual_input"
        if not text:
            return jsonify({"error": "text is required"}), 400
        n = assistant.ingest_text(text=text, source=source)
        return jsonify({"ingested_chunks": n, "source": source}), 200

    @app.post("/ingest/file")
    def ingest_file() -> tuple[Any, int]:
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify({"error": "file is required"}), 400
        ext = Path(f.filename).suffix.lower()
        if ext not in {".txt", ".md"}:
            return jsonify({"error": "only .txt/.md are supported"}), 400
        content = f.read()
        n = assistant.ingest_file(filename=f.filename, file_bytes=content)
        return jsonify({"ingested_chunks": n, "filename": f.filename}), 200

    @app.post("/chat")
    def chat() -> tuple[Any, int]:
        body = request.get_json(force=True, silent=True) or {}
        msg = str(body.get("message", "")).strip()
        if not msg:
            return jsonify({"error": "message is required"}), 400
        out = assistant.chat(msg)
        return jsonify(out), 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
