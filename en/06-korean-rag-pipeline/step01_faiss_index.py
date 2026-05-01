"""Embed documents and build a FAISS index."""

from __future__ import annotations

import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCUMENTS = [
    "Jeju Island is known for tangerines and volcanic landscapes.",
    "The Han River parks are popular for spring walks and cycling.",
    "Busan is well known for seafood markets and beach tourism.",
]
INDEX_PATH = "korean_documents.faiss"


def build_index(documents: list[str]) -> tuple[faiss.IndexFlatIP, int]:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, int(embeddings.shape[1])


def main() -> None:
    index, dimension = build_index(DOCUMENTS)
    faiss.write_index(index, INDEX_PATH)
    print("Built a FAISS index.")
    print(f"Document count: {len(DOCUMENTS)}")
    print(f"Vector dimension: {dimension}")
    print(f"Saved to: {INDEX_PATH}")


if __name__ == "__main__":
    main()
