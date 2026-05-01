"""Groq API로 한국어 요약과 분류를 한 번에 수행합니다."""

from __future__ import annotations

import json
import os

from groq import Groq

MODEL_NAME = "llama-3.1-8b-instant"
ARTICLE = (
    "이 문서는 신규 가입자에게 제공하는 멤버십 혜택을 정리합니다. "
    "연간 요금제를 선택하면 무료 배송, 전용 쿠폰, 조기 할인 행사 접근 권한을 받을 수 있습니다. "
    "단, 일부 해외 배송 상품은 혜택에서 제외됩니다."
)


def require_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 환경 변수가 필요합니다.")
    return api_key


def main() -> None:
    client = Groq(api_key=require_api_key())
    prompt = (
        "다음 한국어 문서를 요약하고 category를 하나 분류해 주세요. "
        "JSON 형식으로만 답하고 키는 summary, category 두 개만 사용하세요.\n\n"
        f"문서:\n{ARTICLE}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "당신은 한국어 문서를 구조화해서 응답하는 도우미입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or '{"summary": "응답 없음", "category": "unknown"}'
    parsed = json.loads(content)
    print("한국어 요약 및 분류 결과")
    print("=" * 40)
    print(f"요약: {parsed['summary']}")
    print(f"분류: {parsed['category']}")


if __name__ == "__main__":
    main()
