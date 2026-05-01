"""Calculate sentence similarity with a KoSimCSE-style replacement model."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTENCES = [
    "The meeting starts at two in the afternoon.",
    "Today's meeting begins at 2 PM.",
    "The lunch menu is soybean stew.",
]


def pairwise_similarity(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(SENTENCES, normalize_embeddings=True)
    scores = pairwise_similarity(embeddings)
    print("Sentence similarity matrix")
    print("=" * 40)
    for row in scores:
        print(" ".join(f"{value:.4f}" for value in row))


if __name__ == "__main__":
    main()
