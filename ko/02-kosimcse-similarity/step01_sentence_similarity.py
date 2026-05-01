"""KoSimCSE 대체 모델로 한국어 문장 유사도를 계산합니다."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTENCES = [
    "오늘 회의는 오후 두 시에 시작합니다.",
    "회의 시작 시간은 오늘 오후 두 시입니다.",
    "점심 메뉴는 된장찌개입니다.",
]


def pairwise_similarity(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T


def main() -> None:
    print(f"모델 로드: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(SENTENCES, normalize_embeddings=True, convert_to_numpy=True)
    scores = pairwise_similarity(embeddings)
    print("문장 유사도 행렬")
    print("=" * 40)
    for row in scores:
        print(" ".join(f"{value:.4f}" for value in row))


if __name__ == "__main__":
    main()
