"""한국어 문서를 임베딩하고 FAISS 인덱스를 생성합니다."""

from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCUMENTS = [
    "제주도는 감귤과 화산 지형으로 유명합니다.",
    "한강공원은 봄철 산책과 자전거 코스로 인기가 많습니다.",
    "부산은 해산물 시장과 해변 관광지로 잘 알려져 있습니다.",
]
INDEX_PATH = "korean_documents.faiss"


def build_index(documents: list[str]) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def main() -> None:
    index, embeddings = build_index(DOCUMENTS)
    faiss.write_index(index, INDEX_PATH)
    print("FAISS 인덱스를 생성했습니다.")
    print(f"문서 수: {len(DOCUMENTS)}")
    print(f"벡터 차원: {embeddings.shape[1]}")
    print(f"저장 경로: {INDEX_PATH}")


if __name__ == "__main__":
    main()
