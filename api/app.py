"""Airbnb guest-concierge retrieval API.

Flow per question:
  1. Embed the question with BGE (same model used at ingest time).
  2. Retrieve the top-4 chunks from Qdrant.
  3. Build a grounded prompt and ask vLLM (OpenAI-compatible endpoint).
  4. Return the answer plus cited sources with similarity scores.

Auth: every route except /health requires ?key=<guest-key>, which must match
the GUEST_KEY env var (injected from the concierge-key Kubernetes secret).
The check-in link itself carries the key, so the link IS the credential.
"""
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

app = FastAPI()

GUEST_KEY = os.environ.get("GUEST_KEY", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
VLLM_URL = os.environ.get("VLLM_URL", "http://vllm-qwen:8000")
COLLECTION = os.environ.get("COLLECTION", "concierge")
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

SYSTEM_PROMPT = """You are the AI concierge for an Airbnb cabin in Idaho City, Idaho.
Answer ONLY from the context chunks below.

- Chunks tagged [cabin_manual] are authoritative house info: quote them directly.
- Chunks tagged [web] are researched local-business info: include phone numbers
  and add a brief "call ahead - hours can be seasonal" hedge.
- If the context does not contain the answer, say you don't have that
  information and suggest contacting the host through Airbnb. Never invent details."""


class Ask(BaseModel):
    question: str


embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
qdrant = QdrantClient(url=QDRANT_URL)


@app.middleware("http")
async def check_key(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if GUEST_KEY and request.query_params.get("key") != GUEST_KEY:
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    return await call_next(request)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    with open("chat.html") as f:
        return f.read()


@app.post("/ask")
def ask(body: Ask):
    vec = embedder.encode(body.question).tolist()
    hits = qdrant.query_points(
        collection_name=COLLECTION, query=vec, limit=4
    ).points

    ctx = "\n\n".join(
        f"[{h.payload.get('source')}] {h.payload.get('section', '')}\n{h.payload.get('text')}"
        for h in hits
    )
    prompt = f"Context:\n{ctx}\n\nGuest question: {body.question}"

    r = httpx.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        },
        timeout=120,
    )
    r.raise_for_status()
    answer = r.json()["choices"][0]["message"]["content"]

    return {
        "answer": answer,
        "sources": [
            {
                "source": h.payload.get("source"),
                "section": h.payload.get("section"),
                "score": round(h.score, 3),
            }
            for h in hits
        ],
    }
