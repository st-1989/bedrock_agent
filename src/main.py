from __future__ import annotations

import argparse
import sys

from bedrock_client import create_chat_model, create_embeddings
from config import load_settings
from pdf_loader import load_and_split_pdfs
from rag import answer_question, build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bedrock + LangChain RAGサンプル")
    parser.add_argument(
        "--question",
        required=True,
        help="PDFに対して質問したい内容",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        settings = load_settings()

        documents = load_and_split_pdfs(
            data_dir=settings.data_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        embeddings = create_embeddings(settings)
        vectorstore = build_index(documents, embeddings)
        chat_model = create_chat_model(settings)

        answer = answer_question(
            question=args.question,
            vectorstore=vectorstore,
            chat_model=chat_model,
            top_k=settings.top_k,
        )

        settings.output_file.parent.mkdir(parents=True, exist_ok=True)
        settings.output_file.write_text(answer, encoding="utf-8")

        print(f"回答を保存しました: {settings.output_file}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"エラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
