"""Show cross-lingual search from a Korean query to English documents."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "서울에서 봄에 가기 좋은 장소"
DOCUMENTS = [
    "The Han River parks are pleasant places to visit during spring in Seoul.",
    "Jeju Island is famous for beaches and summer trips.",
    "Winter hiking in Gangwon Province requires snow gear.",
    "The subway card can be recharged at station kiosks.",
]


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([QUERY], normalize_embeddings=True, convert_to_numpy=True)[0]
    document_embeddings = model.encode(DOCUMENTS, normalize_embeddings=True, convert_to_numpy=True)
    scores = document_embeddings @ query_embedding
    ranking = sorted(zip(DOCUMENTS, scores.tolist()), key=lambda item: item[1], reverse=True)
    print(f"Korean query: {QUERY}")
    print("English document search results")
    print("=" * 60)
    for rank, (document, score) in enumerate(ranking, start=1):
        print(f"{rank}. Score {score:.4f} | {document}")


if __name__ == "__main__":
    main()
