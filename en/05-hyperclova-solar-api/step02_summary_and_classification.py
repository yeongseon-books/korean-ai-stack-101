"""Run summarization and classification with the Groq API."""

from __future__ import annotations

import json
import os

from groq import Groq

MODEL_NAME = "llama-3.1-8b-instant"
ARTICLE = (
    "This document describes the membership benefits offered to new subscribers. "
    "Customers who choose the annual plan receive free shipping, exclusive coupons, and early access to sales. "
    "Some international shipping items are excluded from the offer."
)


def require_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("The GROQ_API_KEY environment variable is required.")
    return api_key


def main() -> None:
    client = Groq(api_key=require_api_key())
    prompt = (
        "Summarize the following document and assign one category. "
        "Respond only as JSON with the keys summary and category.

"
        f"Document:
{ARTICLE}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an assistant that returns structured document outputs."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or '{"summary": "No response", "category": "unknown"}'
    parsed = json.loads(content)
    print("Summarization and classification result")
    print("=" * 40)
    print(f"Summary: {parsed['summary']}")
    print(f"Category: {parsed['category']}")


if __name__ == "__main__":
    main()
