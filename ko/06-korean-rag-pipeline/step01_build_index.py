"""한국어 RAG 인덱스 구성 예제."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.lessons import run_rag_index_demo


if __name__ == "__main__":
    run_rag_index_demo("ko")
