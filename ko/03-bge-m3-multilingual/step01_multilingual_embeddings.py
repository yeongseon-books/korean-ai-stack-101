"""다국어 임베딩 모델로 한국어와 영어 문장을 함께 비교합니다."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCES = [
    "서울의 봄 날씨는 짧고 변덕스럽습니다.",
    "Spring weather in Seoul is short and unpredictable.",
    "제주도는 여름 휴가지로 인기가 많습니다.",
    "Jeju Island is a popular summer destination.",
]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def main() -> None:
    print(f"다국어 모델 로드: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(SENTENCES, normalize_embeddings=True, convert_to_numpy=True)
    print("교차 언어 유사도 비교")
    print("=" * 60)
    for left_index, right_index in combinations(range(len(SENTENCES)), 2):
        score = cosine_similarity(embeddings[left_index], embeddings[right_index])
        print(f"{SENTENCES[left_index]} <> {SENTENCES[right_index]}")
        print(f"유사도: {score:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
