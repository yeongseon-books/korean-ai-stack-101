"""sentence-transformers로 한국어 임베딩을 만들고 코사인 유사도를 비교합니다."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTENCE_PAIRS = [
    ("오늘 서울은 비가 내립니다.", "서울에는 오늘 비가 옵니다."),
    ("점심으로 비빔밥을 먹었습니다.", "아침에 커피를 마셨습니다."),
    ("이 문서는 의료비 청구 절차를 설명합니다.", "의료비 청구 방법을 안내하는 문서입니다."),
]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def format_scores(sentences: Iterable[tuple[str, str]], embeddings: np.ndarray) -> list[str]:
    lines: list[str] = []
    for index, (left, right) in enumerate(sentences):
        score = cosine_similarity(embeddings[index * 2], embeddings[index * 2 + 1])
        lines.extend([
            f"문장 A: {left}",
            f"문장 B: {right}",
            f"코사인 유사도: {score:.4f}",
            "-" * 50,
        ])
    return lines


def main() -> None:
    print(f"모델을 불러옵니다: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    flat_sentences = [sentence for pair in SENTENCE_PAIRS for sentence in pair]
    embeddings = model.encode(flat_sentences, normalize_embeddings=True, convert_to_numpy=True)
    print("한국어 문장 쌍 유사도 비교 결과")
    print("=" * 50)
    for line in format_scores(SENTENCE_PAIRS, embeddings):
        print(line)


if __name__ == "__main__":
    main()
