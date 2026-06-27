from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_and_split_pdfs(
    data_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"PDFフォルダが見つかりません: {data_dir}. data/ フォルダを作成してください。"
        )

    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"data/ にPDFが見つかりませんでした: {data_dir}. PDFファイルを配置してください。"
        )

    pages: list[Document] = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"PDFの読み込みに失敗しました: {pdf_file}") from exc

        for doc in docs:
            doc.metadata["file_name"] = pdf_file.name
        pages.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(pages)
