"""Combine retrieval over documents with Groq answer generation in a RAG pipeline."""

from __future__ import annotations

import os
from typing import Any

import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_NAME = "llama-3.1-8b-instant"
DOCUMENTS = [
    "Spring in Seoul is short, but it offers many cherry blossom walking spots.",
    "Jeju Island combines tangerine experiences with oreum trekking.",
    "Busan is famous for Jagalchi Market and Haeundae Beach.",
]
QUESTION = "Which destination would be good for a spring trip?"


def require_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("The GROQ_API_KEY environment variable is required.")
    return api_key


def retrieve(question: str, documents: list[str], top_k: int = 2) -> list[str]:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    question_embedding = model.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    index: Any = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _, indices = index.search(question_embedding, top_k)
    return [documents[index] for index in indices[0].tolist()]


def generate_answer(question: str, contexts: list[str]) -> str:
    client = Groq(api_key=require_api_key())
    joined_context = "\n".join(f"- {context}" for context in contexts)
    prompt = (
        "Use only the following context to answer in English.\n"
        f"Question: {question}\n"
        f"Context:\n{joined_context}"
    )
    response = client.chat.completions.create(
        model=LLM_NAME,
        messages=[
            {"role": "system", "content": "You answer questions concisely using retrieved context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or "The response was empty."


def main() -> None:
    contexts = retrieve(QUESTION, DOCUMENTS)
    print("Retrieved context")
    print("=" * 40)
    for context in contexts:
        print(f"- {context}")
    print("\nGenerated answer")
    print("=" * 40)
    print(generate_answer(QUESTION, contexts))


if __name__ == "__main__":
    main()
