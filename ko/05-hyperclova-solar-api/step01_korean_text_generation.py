"""Groq API로 한국어 텍스트 생성을 실행합니다."""

from __future__ import annotations

import os

from groq import Groq

MODEL_NAME = "llama-3.1-8b-instant"
SYSTEM_PROMPT = "당신은 한국어 제품 설명을 자연스럽게 작성하는 도우미입니다."
USER_PROMPT = "제주 감귤 탄산수에 대한 3문장 소개를 한국어로 작성해 주세요."


def require_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 환경 변수가 필요합니다.")
    return api_key


def main() -> None:
    client = Groq(api_key=require_api_key())
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.7,
    )
    message = response.choices[0].message.content or "응답이 비어 있습니다."
    print("한국어 텍스트 생성 결과")
    print("=" * 40)
    print(message)


if __name__ == "__main__":
    main()
