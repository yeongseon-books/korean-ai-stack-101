"""Build a small semantic sentence search example."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QUERY = "How do I get a vacation request approved?"
DOCUMENTS = [
    "Submit the vacation request in the HR system and wait for your manager's approval.",
    "Expense receipts must be uploaded before the end of the month.",
    "Please set your chat status to in a meeting.",
    "You can check your remaining leave balance in the payroll portal.",
]


def semantic_search(query: str, documents: list[str], model_name: str) -> list[tuple[str, float]]:
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    document_embeddings = model.encode(documents, normalize_embeddings=True, convert_to_numpy=True)
    scores = document_embeddings @ query_embedding
    ranked = sorted(zip(documents, scores.tolist()), key=lambda item: item[1], reverse=True)
    return ranked


def main() -> None:
    print("Semantic sentence search results")
    print(f"Query: {QUERY}")
    print("=" * 60)
    for rank, (document, score) in enumerate(semantic_search(QUERY, DOCUMENTS, MODEL_NAME), start=1):
        print(f"{rank}. Score {score:.4f} | {document}")


if __name__ == "__main__":
    main()
