"""Mock OCR 결과를 후처리해 구조화된 영수증 데이터로 변환합니다."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

RAW_LINES = [
    "상호: 한빛서점",
    "일시: 2026-05-01 14:30",
    "금액: 18,000원",
    "결제수단: 카드",
]


@dataclass
class Receipt:
    merchant: str
    purchased_at: str
    amount_krw: Decimal
    payment_method: str


def parse_receipt(lines: list[str]) -> Receipt:
    mapping = dict(line.split(": ", 1) for line in lines)
    amount = Decimal(mapping["금액"].replace(",", "").replace("원", ""))
    return Receipt(
        merchant=mapping["상호"],
        purchased_at=mapping["일시"],
        amount_krw=amount,
        payment_method=mapping["결제수단"],
    )


def main() -> None:
    receipt = parse_receipt(RAW_LINES)
    print("구조화된 OCR 결과")
    print("=" * 40)
    print(f"상호: {receipt.merchant}")
    print(f"일시: {receipt.purchased_at}")
    print(f"금액: {receipt.amount_krw}원")
    print(f"결제수단: {receipt.payment_method}")


if __name__ == "__main__":
    main()
