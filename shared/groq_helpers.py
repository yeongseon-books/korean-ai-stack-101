"""Groq helper with a fallback path for offline runs."""

from __future__ import annotations

import os

from groq import Groq

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def complete_with_groq(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 300,
    model: str = DEFAULT_GROQ_MODEL,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "GROQ_API_KEY is not set. This fallback response shows where the real Groq answer will appear."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""
