"""한국어 유사 문장 검색 시스템을 간단히 구현합니다."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QUERY = "휴가 신청을 승인받으려면 어떻게 해야 하나요?"
DOCUMENTS = [
    "휴가 신청은 인사 시스템에서 작성한 뒤 팀장 승인 절차를 거칩니다.",
    "비용 정산 영수증은 월말까지 업로드해야 합니다.",
    "사내 메신저 상태는 회의 중으로 변경해 주세요.",
    "연차 잔여일은 급여 명세서에서 확인할 수 있습니다.",
]


def semantic_search(query: str, documents: list[str], model_name: str) -> list[tuple[str, float]]:
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    document_embeddings = model.encode(documents, normalize_embeddings=True)
    scores = document_embeddings @ query_embedding
    ranked = sorted(zip(documents, scores.tolist()), key=lambda item: item[1], reverse=True)
    return ranked


def main() -> None:
    print("한국어 유사 문장 검색 결과")
    print(f"질의: {QUERY}")
    print("=" * 60)
    for rank, (document, score) in enumerate(semantic_search(QUERY, DOCUMENTS, MODEL_NAME), start=1):
        print(f"{rank}. 점수 {score:.4f} | {document}")


if __name__ == "__main__":
    main()
