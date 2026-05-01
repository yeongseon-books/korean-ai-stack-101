# korean-ai-stack-101

Step-by-step Korean AI stack examples in Korean and English.

## Structure

- `ko/`: Korean examples with Korean prompts, corpus text, and terminal output
- `en/`: English examples with the same logic and English strings
- Each lesson has two executable steps that can run independently

## Lessons

1. Korean embedding model comparison
2. Korean sentence similarity
3. Multilingual embedding search
4. OCR pipeline with a mock CLOVA OCR implementation
5. Korean LLM API examples powered by Groq (`llama-3.1-8b-instant`)
6. Korean RAG pipeline with FAISS

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your-groq-api-key"
```

## Notes

- Services that require separate API keys such as CLOVA OCR, HyperCLOVA X, and Kakao KoGPT are replaced with mock implementations.
- KoSimCSE and BGE-M3 examples use `sentence-transformers` models that work locally.
- Solar API examples are replaced with Groq chat completions.
- Every Python file is designed to be directly executable.
