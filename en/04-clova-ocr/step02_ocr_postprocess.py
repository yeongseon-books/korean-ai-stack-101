"""Post-process mock OCR output into a structured receipt record."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

RAW_LINES = [
    "store: Hanbit Books",
    "timestamp: 2026-05-01 14:30",
    "amount: 18,000 KRW",
    "payment_method: card",
]


@dataclass
class Receipt:
    merchant: str
    purchased_at: str
    amount_krw: Decimal
    payment_method: str


def parse_receipt(lines: list[str]) -> Receipt:
    mapping = dict(line.split(": ", 1) for line in lines)
    amount = Decimal(mapping["amount"].replace(",", "").replace(" KRW", ""))
    return Receipt(
        merchant=mapping["store"],
        purchased_at=mapping["timestamp"],
        amount_krw=amount,
        payment_method=mapping["payment_method"],
    )


def main() -> None:
    receipt = parse_receipt(RAW_LINES)
    print("Structured OCR output")
    print("=" * 40)
    print(f"Store: {receipt.merchant}")
    print(f"Timestamp: {receipt.purchased_at}")
    print(f"Amount: {receipt.amount_krw} KRW")
    print(f"Payment method: {receipt.payment_method}")


if __name__ == "__main__":
    main()
