"""Chunk the local-business research markdown by ## section, embed, and upsert.

Run the same way as ingest_cabin.py (Qdrant reachable on localhost:6333).
Uses upsert (not recreate) so the cabin chunks stay intact.
IDs continue after the cabin chunk range to avoid collisions.
"""
import re
from datetime import date
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

MD_PATH = "../data/idaho-city-business-guide.md"
COLLECTION = "concierge"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
ID_OFFSET = 1000  # keep clear of cabin-chunk IDs


def chunk_markdown(path):
    with open(path) as f:
        text = f.read()
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    # parts[0] is preamble; then alternating title/body
    chunks = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            chunks.append((title, body))
    return chunks


def main():
    chunks = chunk_markdown(MD_PATH)
    print(f"{len(chunks)} web sections from {MD_PATH}")

    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)
    retrieved = date.today().isoformat()

    points = [
        PointStruct(
            id=ID_OFFSET + i,
            vector=model.encode(text).tolist(),
            payload={
                "source": "web",
                "section": title,
                "text": text,
                "retrieved_at": retrieved,
            },
        )
        for i, (title, text) in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"upserted {len(points)} web chunks")


if __name__ == "__main__":
    main()
