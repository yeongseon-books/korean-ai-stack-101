"""CLOVA OCR을 대신하는 mock OCR 파이프라인을 구현합니다."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OcrField:
    name: str
    text: str
    confidence: float


@dataclass
class OcrResult:
    image_path: str
    fields: list[OcrField]


class MockClovaOcrClient:
    """샘플 영수증 텍스트를 반환하는 mock 클라이언트입니다."""

    def extract_text(self, image_path: str) -> OcrResult:
        fields = [
            OcrField(name="상호", text="한빛서점", confidence=0.99),
            OcrField(name="일시", text="2026-05-01 14:30", confidence=0.98),
            OcrField(name="금액", text="18,000원", confidence=0.97),
        ]
        return OcrResult(image_path=image_path, fields=fields)


def main() -> None:
    client = MockClovaOcrClient()
    result = client.extract_text("sample_receipt.png")
    print("Mock OCR 추출 결과")
    print("=" * 40)
    print(f"이미지: {result.image_path}")
    for field in result.fields:
        print(f"{field.name}: {field.text} (신뢰도 {field.confidence:.2f})")


if __name__ == "__main__":
    main()
