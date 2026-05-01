"""여러 sentence-transformers 모델로 한국어 문장 검색 성능을 간단히 비교합니다."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

MODELS = {
    "KoSimCSE 대체": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BGE-M3 대체": "sentence-transformers/all-MiniLM-L6-v2",
    "Solar 대체": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
QUERY = "환불 정책을 설명한 문서"
CORPUS = [
    "환불 요청은 구매 후 7일 이내에 접수해야 합니다.",
    "모든 직원은 분기마다 보안 교육을 수료해야 합니다.",
    "고객 지원 센터 운영 시간은 평일 오전 9시부터 오후 6시까지입니다.",
    "문서에는 주문 취소와 환불 절차가 정리되어 있습니다.",
]
EXPECTED_INDEX = 3


@dataclass
class BenchmarkResult:
    label: str
    best_index: int
    best_score: float
    latency_ms: float


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix @ vector


def benchmark_model(label: str, model_name: str) -> BenchmarkResult:
    started = time.perf_counter()
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([QUERY], normalize_embeddings=True)[0]
    corpus_embeddings = model.encode(CORPUS, normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, corpus_embeddings)
    latency_ms = (time.perf_counter() - started) * 1000
    best_index = int(np.argmax(scores))
    return BenchmarkResult(label=label, best_index=best_index, best_score=float(scores[best_index]), latency_ms=latency_ms)


def main() -> None:
    print("한국어 임베딩 모델 간단 벤치마크")
    print(f"질의: {QUERY}")
    print("=" * 70)
    for label, model_name in MODELS.items():
        result = benchmark_model(label, model_name)
        verdict = "Pass" if result.best_index == EXPECTED_INDEX else "Check"
        print(f"모델: {label}")
        print(f"선택 문서: {CORPUS[result.best_index]}")
        print(f"점수: {result.best_score:.4f}")
        print(f"지연 시간: {result.latency_ms:.1f}ms")
        print(f"판정: {verdict}")
        print("-" * 70)


if __name__ == "__main__":
    main()
