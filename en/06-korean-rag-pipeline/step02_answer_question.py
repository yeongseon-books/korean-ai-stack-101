"""Korean RAG answer generation example."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.lessons import run_rag_answer_demo


if __name__ == "__main__":
    run_rag_answer_demo("en")
