"""Generate text with the Groq API."""

from __future__ import annotations

import os

from groq import Groq

MODEL_NAME = "llama-3.1-8b-instant"
SYSTEM_PROMPT = "You write concise product descriptions in natural English."
USER_PROMPT = "Write a three-sentence introduction for a sparkling tangerine drink from Jeju."


def require_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("The GROQ_API_KEY environment variable is required.")
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
    message = response.choices[0].message.content or "The response was empty."
    print("Text generation result")
    print("=" * 40)
    print(message)


if __name__ == "__main__":
    main()
