"""Create embeddings with sentence-transformers and compare cosine similarity."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTENCE_PAIRS = [
    ("It is raining in Seoul today.", "Seoul is getting rain today."),
    ("I had bibimbap for lunch.", "I drank coffee this morning."),
    ("This document explains how to submit a medical claim.", "This guide describes the medical claim process."),
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
            f"Sentence A: {left}",
            f"Sentence B: {right}",
            f"Cosine similarity: {score:.4f}",
            "-" * 50,
        ])
    return lines


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    flat_sentences = [sentence for pair in SENTENCE_PAIRS for sentence in pair]
    embeddings = model.encode(flat_sentences, normalize_embeddings=True, convert_to_numpy=True)
    print("Similarity results for sentence pairs")
    print("=" * 50)
    for line in format_scores(SENTENCE_PAIRS, embeddings):
        print(line)


if __name__ == "__main__":
    main()
