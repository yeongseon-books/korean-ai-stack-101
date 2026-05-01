"""Mock clients that keep stable API-like signatures."""

from __future__ import annotations

import time
import uuid


def call_clova_ocr(
    api_url: str,
    secret_key: str,
    request_json: dict,
    image_bytes: bytes | None = None,
    *,
    timeout: int = 30,
) -> dict:
    """Mock wrapper shaped like a typical CLOVA OCR HTTP helper."""
    image_name = request_json.get("images", [{}])[0].get("name", "mock-image")
    language = request_json.get("metadata", {}).get("language", "ko")
    lines = request_json.get("metadata", {}).get("mock_lines", [])
    fields = []
    for index, line in enumerate(lines, start=1):
        fields.append(
            {
                "inferText": line,
                "inferConfidence": 0.99,
                "type": "LINE",
                "lineNo": index,
            }
        )

    return {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "images": [
            {
                "uid": str(uuid.uuid4()),
                "name": image_name,
                "inferResult": "SUCCESS",
                "message": f"mock response from {api_url}",
                "validationResult": {"result": "SUCCESS"},
                "metadata": {
                    "language": language,
                    "timeout": timeout,
                    "secret_prefix": secret_key[:4],
                    "has_image_bytes": image_bytes is not None,
                },
                "fields": fields,
            }
        ],
    }


class HyperClovaXMockClient:
    """Mock chat client with an API shape similar to a hosted completion endpoint."""

    def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.5,
        top_p: float = 0.8,
        top_k: int = 0,
        repeat_penalty: float = 1.1,
        stop_before: list[str] | None = None,
        include_ai_filters: bool = True,
        seed: int = 0,
    ) -> dict:
        user_message = next((item["content"] for item in reversed(messages) if item.get("role") == "user"), "")
        language = "ko" if any("가" <= ch <= "힣" for ch in user_message) else "en"
        if language == "ko":
            content = (
                "KoSimCSE는 한국어 문장 유사도에 집중한 경량 선택지입니다. "
                "BGE-M3는 한국어와 영어가 섞인 검색에 더 유연합니다. "
                "운영 전에는 실제 데이터 분포로 두 모델을 함께 평가하는 편이 안전합니다."
            )
        else:
            content = (
                "KoSimCSE is a focused option for Korean sentence similarity. "
                "BGE-M3 is more flexible for mixed Korean and English retrieval. "
                "Before production, evaluate both against your own data distribution."
            )

        return {
            "id": f"mock-hcx-{uuid.uuid4()}",
            "model": model,
            "result": {
                "message": {"role": "assistant", "content": content},
                "stopReason": "length" if max_tokens < 128 else "stop",
                "inputLength": sum(len(item.get("content", "")) for item in messages),
                "outputLength": min(len(content), max_tokens),
                "seed": seed,
                "topP": top_p,
                "topK": top_k,
                "temperature": temperature,
                "repeatPenalty": repeat_penalty,
                "stopBefore": stop_before or [],
            },
            "aiFilter": [
                {
                    "groupName": "safety",
                    "name": "mock-safe",
                    "score": "0",
                }
            ]
            if include_ai_filters
            else [],
        }
