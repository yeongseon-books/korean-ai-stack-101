"""Static sample data for Korean and English lessons."""

LESSON_TEXT = {
    "ko": {
        "language_name": "한국어",
        "sentence_pairs": [
            ("서울시청 근처에서 점심을 먹었다.", "서울 시청 주변에서 식사를 했다.", "유사"),
            ("벡터 검색은 의미 기반 검색에 적합하다.", "FAISS는 임베딩 검색 속도를 높여 준다.", "관련"),
            ("제주도는 오늘 비가 온다.", "GPU 메모리 사용량이 급증했다.", "무관"),
            ("HyperCLOVA X는 한국어 생성 품질이 강점이다.", "네이버 모델은 한국어 응답에 최적화되어 있다.", "유사"),
        ],
        "documents": [
            "KoSimCSE는 한국어 문장 유사도 태스크에 자주 쓰이는 임베딩 모델이다.",
            "BGE-M3는 한국어와 영어가 섞인 문서 검색에 잘 맞는다.",
            "CLOVA OCR은 영수증, 계약서, 명함 같은 문서에서 텍스트를 추출한다.",
            "HyperCLOVA X는 한국어 질의응답과 요약 시나리오에 자주 언급된다.",
            "FAISS는 임베딩 벡터를 빠르게 검색하기 위한 라이브러리다.",
            "Groq API는 빠른 응답 속도로 LLM 프로토타입을 만들 때 편리하다.",
        ],
        "query": "한국어 문장 유사도와 벡터 검색에 맞는 도구를 찾고 싶다.",
        "multilingual_docs": [
            "한국어 FAQ 데이터는 KoSimCSE로 빠르게 실험할 수 있다.",
            "BGE-M3 works well when Korean tickets include English product names.",
            "Solar Embedding focuses on Korean and English retrieval quality.",
            "한국어 고객센터 문서에도 API, SDK 같은 영어 용어가 자주 섞인다.",
        ],
        "multilingual_queries": [
            "영어 제품명이 섞인 한국어 문의를 검색하고 싶다.",
            "Find Korean support articles that mention SDK setup.",
        ],
        "ocr_lines": [
            "주문번호: A-1024",
            "상품명: 한국어 OCR 데모 책",
            "총액: 18,000원",
        ],
        "clova_prompt": "영수증에서 주문번호와 총액을 읽어 주세요.",
        "hyperclova_messages": [
            {"role": "system", "content": "당신은 한국어 AI 스택 튜터입니다."},
            {"role": "user", "content": "KoSimCSE와 BGE-M3의 차이를 3문장으로 설명해 주세요."},
        ],
        "groq_prompt": "한국어 AI 스택 입문자가 KoSimCSE, BGE-M3, OCR mock, RAG를 왜 함께 배우면 좋은지 4문장으로 설명해 주세요.",
        "rag_question": "한국어 고객지원 문서를 위한 RAG를 시작할 때 어떤 구성으로 출발하면 좋을까?",
    },
    "en": {
        "language_name": "English",
        "sentence_pairs": [
            ("I tested Korean sentence search with FAISS.", "I built a vector search demo for Korean text.", "similar"),
            ("BGE-M3 supports multilingual retrieval.", "Mixed Korean and English tickets are easier to search.", "related"),
            ("The OCR pipeline extracted invoice fields.", "The GPU temperature crossed eighty degrees.", "unrelated"),
            ("HyperCLOVA X is often discussed for Korean generation.", "NAVER models are tuned for Korean responses.", "similar"),
        ],
        "documents": [
            "KoSimCSE is commonly used for Korean sentence similarity experiments.",
            "BGE-M3 is a strong fit for search over mixed Korean and English documents.",
            "CLOVA OCR extracts text from receipts, contracts, and business cards.",
            "HyperCLOVA X is frequently mentioned in Korean summarization and Q&A workflows.",
            "FAISS is a library for fast similarity search over embedding vectors.",
            "Groq API is useful when you want a fast LLM prototype with minimal setup.",
        ],
        "query": "I want tools for Korean sentence similarity and vector retrieval.",
        "multilingual_docs": [
            "Korean FAQ datasets are easy to prototype with KoSimCSE-style workflows.",
            "BGE-M3 helps when Korean support tickets include English product names.",
            "Solar Embedding focuses on bilingual Korean and English retrieval quality.",
            "Korean support content often mixes API, SDK, and deployment vocabulary.",
        ],
        "multilingual_queries": [
            "Search Korean issues that contain English product names.",
            "한국어 문서에서 SDK 설치 가이드를 찾고 싶다.",
        ],
        "ocr_lines": [
            "Order ID: A-1024",
            "Item: Korean OCR demo book",
            "Total: 18,000 KRW",
        ],
        "clova_prompt": "Read the order id and total from this receipt.",
        "hyperclova_messages": [
            {"role": "system", "content": "You are a Korean AI stack tutor."},
            {"role": "user", "content": "Explain the difference between KoSimCSE and BGE-M3 in three sentences."},
        ],
        "groq_prompt": "Explain in four sentences why a beginner should learn Korean embeddings, OCR mocks, and RAG together.",
        "rag_question": "What is a practical starting architecture for RAG over Korean customer support content?",
    },
}
