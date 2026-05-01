"""Embedding and FAISS helpers used by both language tracks."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

SUBSTITUTE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ORIGINAL_MODEL_NAMES = {
    "KoSimCSE": "BM-K/KoSimCSE-roberta-multitask",
    "BGE-M3": "BAAI/bge-m3",
    "Solar Embedding": "upstage/solar-embedding-1-large-query",
}
MODEL_DIMENSIONS = {
    "KoSimCSE": 768,
    "BGE-M3": 1024,
    "Solar Embedding": 4096,
}


@lru_cache(maxsize=1)
def load_demo_model() -> SentenceTransformer:
    return SentenceTransformer(SUBSTITUTE_MODEL_NAME)


def encode_texts(texts: Iterable[str]) -> np.ndarray:
    model = load_demo_model()
    vectors = model.encode(list(texts), normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def simulated_model_score(model_name: str, text_a: str, text_b: str) -> float:
    vectors = encode_texts([text_a, text_b])
    base_score = cosine_similarity(vectors[0], vectors[1])

    if model_name == "KoSimCSE":
        bonus = 0.04 if all(ord(ch) < 128 or "가" <= ch <= "힣" for ch in f"{text_a}{text_b}") else 0.02
    elif model_name == "BGE-M3":
        bonus = 0.05 if any(token in f"{text_a} {text_b}" for token in ["API", "SDK", "Python", "English"]) else 0.03
    else:
        bonus = 0.06

    return min(0.995, max(-0.995, base_score + bonus))


def simulate_dimension(vectors: np.ndarray, target_dim: int) -> np.ndarray:
    current_dim = vectors.shape[1]
    if target_dim == current_dim:
        return vectors.astype(np.float32)
    if target_dim < current_dim:
        return vectors[:, :target_dim].astype(np.float32)

    repeats = (target_dim + current_dim - 1) // current_dim
    tiled = np.tile(vectors, (1, repeats))[:, :target_dim]
    return tiled.astype(np.float32)


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])
    cast(Any, index).add(vectors.astype(np.float32))
    return index
