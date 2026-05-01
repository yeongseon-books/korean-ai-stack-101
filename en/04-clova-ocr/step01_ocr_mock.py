"""Implement a mock OCR pipeline that stands in for CLOVA OCR."""

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
    """Mock client that returns sample receipt text."""

    def extract_text(self, image_path: str) -> OcrResult:
        fields = [
            OcrField(name="store", text="Hanbit Books", confidence=0.99),
            OcrField(name="timestamp", text="2026-05-01 14:30", confidence=0.98),
            OcrField(name="amount", text="18,000 KRW", confidence=0.97),
        ]
        return OcrResult(image_path=image_path, fields=fields)


def main() -> None:
    client = MockClovaOcrClient()
    result = client.extract_text("sample_receipt.png")
    print("Mock OCR extraction result")
    print("=" * 40)
    print(f"Image: {result.image_path}")
    for field in result.fields:
        print(f"{field.name}: {field.text} (confidence {field.confidence:.2f})")


if __name__ == "__main__":
    main()
