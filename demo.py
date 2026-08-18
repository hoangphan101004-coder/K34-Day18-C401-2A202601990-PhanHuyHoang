"""Interactive demo for the completed production RAG pipeline."""

import argparse
import sys

from src.pipeline import build_pipeline, run_query


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Demo hỏi đáp Production RAG")
    parser.add_argument("--question", "-q", help="Câu hỏi chạy một lần")
    parser.add_argument("--rebuild", action="store_true", help="Tạo lại dense index trong Qdrant")
    parser.add_argument("--show-context", action="store_true", help="Hiện các context được truy xuất")
    args = parser.parse_args()

    search, reranker = build_pipeline(rebuild_dense=args.rebuild)

    def ask(question: str) -> None:
        answer, contexts = run_query(question, search, reranker)
        print(f"\nTrả lời: {answer}")
        if args.show_context:
            print("\nContext:")
            for index, context in enumerate(contexts, start=1):
                print(f"[{index}] {context}")

    if args.question:
        ask(args.question)
        return

    print("\nNhập câu hỏi; để trống hoặc gõ 'exit' để kết thúc.")
    while True:
        try:
            question = input("\nCâu hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        ask(question)


if __name__ == "__main__":
    main()
