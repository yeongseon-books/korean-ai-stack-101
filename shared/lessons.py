"""Lesson runners shared by ko/ and en/ scripts."""

from __future__ import annotations

from textwrap import indent
from typing import Any, cast

import numpy as np

from shared.data import LESSON_TEXT
from shared.embedding_helpers import (
    MODEL_DIMENSIONS,
    ORIGINAL_MODEL_NAMES,
    SUBSTITUTE_MODEL_NAME,
    build_index,
    encode_texts,
    simulated_model_score,
    simulate_dimension,
)
from shared.groq_helpers import complete_with_groq
from shared.mock_clients import HyperClovaXMockClient, call_clova_ocr


def _print_header(language: str, title: str) -> None:
    prefix = "실행 주제" if language == "ko" else "Topic"
    print(f"{prefix}: {title}")
    print("-" * 72)


def run_embedding_model_compare(language: str) -> None:
    data = LESSON_TEXT[language]
    _print_header(language, "Korean embedding model comparison")
    if language == "ko":
        print(f"대체 런타임 모델: {SUBSTITUTE_MODEL_NAME}")
        print("원본 모델명은 주석과 출력에 함께 남겨 둡니다.")
    else:
        print(f"Runtime substitute model: {SUBSTITUTE_MODEL_NAME}")
        print("Original model names are still shown for study notes.")
    print()

    for model_name, original_name in ORIGINAL_MODEL_NAMES.items():
        if language == "ko":
            print(f"[{model_name}] 원본 모델: {original_name}")
        else:
            print(f"[{model_name}] original model: {original_name}")
        for text_a, text_b, label in data["sentence_pairs"]:
            score = simulated_model_score(model_name, text_a, text_b)
            print(f"- {label:>9} | {score:.3f} | {text_a} <> {text_b}")
        print()


def run_embedding_dimension_demo(language: str) -> None:
    data = LESSON_TEXT[language]
    _print_header(language, "Dimension and retrieval trade-offs")
    vectors = encode_texts(data["documents"])
    query_vector = encode_texts([data["query"]])

    for model_name, target_dim in MODEL_DIMENSIONS.items():
        projected = simulate_dimension(vectors, target_dim)
        index = build_index(projected)
        query_projected = simulate_dimension(query_vector, target_dim)
        distances, indices = cast(Any, index).search(query_projected, k=2)
        memory_kb = projected.nbytes / 1024
        top_doc = data["documents"][int(indices[0][0])]
        if language == "ko":
            print(f"{model_name:16} | 차원={target_dim:4d} | 메모리={memory_kb:7.1f}KB | 최고 점수={distances[0][0]:.3f}")
            print(f"  상위 문서: {top_doc}")
        else:
            print(f"{model_name:16} | dims={target_dim:4d} | memory={memory_kb:7.1f}KB | top score={distances[0][0]:.3f}")
            print(f"  top document: {top_doc}")


def run_similarity_scores(language: str) -> None:
    data = LESSON_TEXT[language]
    _print_header(language, "Sentence similarity with a KoSimCSE-style workflow")
    print(f"Using substitute model: {SUBSTITUTE_MODEL_NAME}  # original: {ORIGINAL_MODEL_NAMES['KoSimCSE']}")
    for text_a, text_b, label in data["sentence_pairs"]:
        vectors = encode_texts([text_a, text_b])
        score = float(np.dot(vectors[0], vectors[1]))
        print(f"- {label:>9} | {score:.3f} | {text_a} <> {text_b}")


def run_similarity_search(language: str) -> None:
    data = LESSON_TEXT[language]
    _print_header(language, "Similarity search over Korean-focused documents")
    vectors = encode_texts(data["documents"])
    index = build_index(vectors)
    query_vector = encode_texts([data["query"]])
    distances, indices = cast(Any, index).search(query_vector, k=3)
    for rank, (distance, doc_index) in enumerate(zip(distances[0], indices[0]), start=1):
        if language == "ko":
            print(f"{rank}. 점수={distance:.3f} | {data['documents'][int(doc_index)]}")
        else:
            print(f"{rank}. score={distance:.3f} | {data['documents'][int(doc_index)]}")


def run_multilingual_demo(language: str) -> None:
    data = LESSON_TEXT[language]
    _print_header(language, "Multilingual retrieval with a BGE-M3-style workflow")
    print(f"Using substitute model: {SUBSTITUTE_MODEL_NAME}  # original: {ORIGINAL_MODEL_NAMES['BGE-M3']}")
    vectors = encode_texts(data["multilingual_docs"])
    index = build_index(vectors)
    for query in data["multilingual_queries"]:
        query_vector = encode_texts([query])
        distances, indices = cast(Any, index).search(query_vector, k=2)
        if language == "ko":
            print(f"질의: {query}")
        else:
            print(f"Query: {query}")
        for rank, (distance, doc_index) in enumerate(zip(distances[0], indices[0]), start=1):
            print(f"  {rank}. {distance:.3f} | {data['multilingual_docs'][int(doc_index)]}")


def run_clova_ocr_demo(language: str) -> None:
    data = LESSON_TEXT[language]
    _print_header(language, "CLOVA OCR mock client")
    request_json = {
        "images": [{"format": "png", "name": "receipt-sample"}],
        "requestId": "demo-request",
        "version": "V2",
        "timestamp": 1714546800000,
        "metadata": {
            "language": language,
            "prompt": data["clova_prompt"],
            "mock_lines": data["ocr_lines"],
        },
    }
    response = call_clova_ocr(
        "https://mock-clova-ocr.local/custom/v1/0000/general",
        "mock-secret-key",
        request_json,
        None,
        timeout=15,
    )
    lines = [field["inferText"] for field in response["images"][0]["fields"]]
    if language == "ko":
        print("추출 결과:")
    else:
        print("Extracted lines:")
    print(indent("\n".join(lines), prefix="  "))


def run_hyperclova_mock_demo(language: str) -> None:
    data = LESSON_TEXT[language]
    _print_header(language, "HyperCLOVA X mock chat completion")
    client = HyperClovaXMockClient()
    response = client.create_chat_completion(
        model="HCX-005",
        messages=data["hyperclova_messages"],
        max_tokens=200,
        temperature=0.2,
        top_p=0.8,
        top_k=0,
        repeat_penalty=1.1,
        stop_before=["<END>"],
        include_ai_filters=True,
        seed=7,
    )
    content = response["result"]["message"]["content"]
    if language == "ko":
        print("모의 응답:")
    else:
        print("Mock response:")
    print(indent(content, prefix="  "))


def run_groq_replacement_demo(language: str) -> None:
    data = LESSON_TEXT[language]
    _print_header(language, "Groq replacement for hosted Korean generation demos")
    system_prompt = (
        "당신은 한국어 AI 스택을 설명하는 튜터입니다. 한국어로 답하세요."
        if language == "ko"
        else "You are a tutor explaining the Korean AI stack. Answer in English."
    )
    answer = complete_with_groq(
        system_prompt=system_prompt,
        user_prompt=data["groq_prompt"],
        temperature=0.3,
        max_tokens=220,
    )
    print(indent(answer.strip(), prefix="  "))


def run_rag_index_demo(language: str) -> None:
    data = LESSON_TEXT[language]
    _print_header(language, "Build a small Korean RAG index")
    vectors = encode_texts(data["documents"])
    index = build_index(vectors)
    if language == "ko":
        print(f"문서 수: {len(data['documents'])}")
        print(f"벡터 차원: {vectors.shape[1]}")
        print(f"FAISS ntotal: {index.ntotal}")
    else:
        print(f"Document count: {len(data['documents'])}")
        print(f"Vector dimension: {vectors.shape[1]}")
        print(f"FAISS ntotal: {index.ntotal}")


def run_rag_answer_demo(language: str) -> None:
    data = LESSON_TEXT[language]
    _print_header(language, "Answer a question with retrieval-augmented context")
    vectors = encode_texts(data["documents"])
    index = build_index(vectors)
    query_vector = encode_texts([data["rag_question"]])
    distances, indices = cast(Any, index).search(query_vector, k=3)
    context_lines = [data["documents"][int(doc_index)] for doc_index in indices[0]]
    if language == "ko":
        system_prompt = "당신은 한국어 RAG 설계를 설명하는 시니어 엔지니어입니다."
        user_prompt = (
            f"질문: {data['rag_question']}\n"
            f"검색 문맥:\n- " + "\n- ".join(context_lines) + "\n"
            "문맥을 활용해 실무적인 시작 구성을 4문장으로 설명해 주세요."
        )
    else:
        system_prompt = "You are a senior engineer explaining Korean RAG design."
        user_prompt = (
            f"Question: {data['rag_question']}\n"
            f"Retrieved context:\n- " + "\n- ".join(context_lines) + "\n"
            "Use the context to explain a practical starting architecture in four sentences."
        )
    answer = complete_with_groq(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=240,
    )
    context_label = "검색 문맥" if language == "ko" else "Retrieved context"
    answer_label = "생성 답변" if language == "ko" else "Generated answer"
    print(f"{context_label}:")
    print(indent("\n".join(f"- {line} ({score:.3f})" for line, score in zip(context_lines, distances[0])), prefix="  "))
    print(f"\n{answer_label}:")
    print(indent(answer.strip(), prefix="  "))
