"""Chunk the cabin welcome guide (.docx) by section, embed, and load into Qdrant.

Run from a machine that can reach Qdrant (e.g. EC2 host with a port-forward):
    kubectl port-forward svc/qdrant 6333:6333 &
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python ingest_cabin.py

The .docx is intentionally NOT in git (wifi password + personal details).
Place it next to this script before running.
"""
from docx import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DOCX_PATH = "Airbnb_Cabin_Welcome___House_Guide.docx"
COLLECTION = "concierge"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def chunk_by_section(doc):
    """One chunk per heading section: heading title + following paragraphs."""
    chunks, title, buf = [], "General", []
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading"):
            if buf:
                chunks.append((title, "\n".join(buf)))
            title, buf = p.text.strip(), []
        elif p.text.strip():
            buf.append(p.text.strip())
    if buf:
        chunks.append((title, "\n".join(buf)))
    return [(t, b) for t, b in chunks if b]


def main():
    doc = Document(DOCX_PATH)
    chunks = chunk_by_section(doc)
    print(f"{len(chunks)} sections from {DOCX_PATH}")

    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=model.encode(text).tolist(),
            payload={"source": "cabin_manual", "section": title, "text": text},
        )
        for i, (title, text) in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"upserted {len(points)} cabin chunks")


if __name__ == "__main__":
    main()
