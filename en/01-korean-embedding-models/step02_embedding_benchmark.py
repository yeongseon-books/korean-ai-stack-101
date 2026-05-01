"""Run a small retrieval benchmark across multiple sentence-transformers models."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

MODELS = {
    "KoSimCSE replacement": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BGE-M3 replacement": "sentence-transformers/all-MiniLM-L6-v2",
    "Solar replacement": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
QUERY = "document that explains the refund policy"
CORPUS = [
    "Refund requests must be submitted within seven days of purchase.",
    "Every employee must complete security training each quarter.",
    "Customer support is available from 9 AM to 6 PM on weekdays.",
    "This document summarizes the cancellation and refund process.",
]
EXPECTED_INDEX = 3


@dataclass
class BenchmarkResult:
    label: str
    best_index: int
    best_score: float
    latency_ms: float


def score_documents(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix @ vector


def benchmark_model(label: str, model_name: str) -> BenchmarkResult:
    started = time.perf_counter()
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([QUERY], normalize_embeddings=True)[0]
    corpus_embeddings = model.encode(CORPUS, normalize_embeddings=True)
    scores = score_documents(query_embedding, corpus_embeddings)
    latency_ms = (time.perf_counter() - started) * 1000
    best_index = int(np.argmax(scores))
    return BenchmarkResult(label=label, best_index=best_index, best_score=float(scores[best_index]), latency_ms=latency_ms)


def main() -> None:
    print("Small benchmark for Korean-oriented embedding models")
    print(f"Query: {QUERY}")
    print("=" * 70)
    for label, model_name in MODELS.items():
        result = benchmark_model(label, model_name)
        verdict = "Pass" if result.best_index == EXPECTED_INDEX else "Check"
        print(f"Model: {label}")
        print(f"Selected document: {CORPUS[result.best_index]}")
        print(f"Score: {result.best_score:.4f}")
        print(f"Latency: {result.latency_ms:.1f}ms")
        print(f"Verdict: {verdict}")
        print("-" * 70)


if __name__ == "__main__":
    main()
