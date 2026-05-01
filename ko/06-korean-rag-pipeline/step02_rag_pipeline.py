"""한국어 문서 검색과 Groq 답변 생성을 결합한 RAG 파이프라인입니다."""

from __future__ import annotations

import os

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_NAME = "llama-3.1-8b-instant"
DOCUMENTS = [
    "서울의 봄은 짧지만 벚꽃 명소가 많아 산책하기 좋습니다.",
    "제주도는 감귤 체험과 오름 트레킹을 함께 즐길 수 있습니다.",
    "부산은 자갈치시장과 해운대 해변이 대표 관광지입니다.",
]
QUESTION = "봄 여행지로 어디가 좋을까요?"


def require_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 환경 변수가 필요합니다.")
    return api_key


def retrieve(question: str, documents: list[str], top_k: int = 2) -> list[str]:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents, normalize_embeddings=True).astype("float32")
    question_embedding = model.encode([question], normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _, indices = index.search(question_embedding, top_k)
    return [documents[index] for index in indices[0].tolist()]


def generate_answer(question: str, contexts: list[str]) -> str:
    client = Groq(api_key=require_api_key())
    joined_context = "
".join(f"- {context}" for context in contexts)
    prompt = (
        "다음 참고 문맥만 사용해 한국어로 답변해 주세요.
"
        f"질문: {question}
"
        f"문맥:
{joined_context}"
    )
    response = client.chat.completions.create(
        model=LLM_NAME,
        messages=[
            {"role": "system", "content": "당신은 검색 기반 답변을 간결하게 작성하는 도우미입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or "응답이 비어 있습니다."


def main() -> None:
    contexts = retrieve(QUESTION, DOCUMENTS)
    print("검색된 문맥")
    print("=" * 40)
    for context in contexts:
        print(f"- {context}")
    print("
생성된 답변")
    print("=" * 40)
    print(generate_answer(QUESTION, contexts))


if __name__ == "__main__":
    main()
