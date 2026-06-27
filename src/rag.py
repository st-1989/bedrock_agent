from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def _format_page_number(metadata: dict) -> str:
    page = metadata.get("page")
    if isinstance(page, int):
        return str(page + 1)
    return "不明"


def build_index(documents: list[Document], embeddings) -> FAISS:
    if not documents:
        raise ValueError("インデックス化対象のドキュメントがありません。")
    return FAISS.from_documents(documents, embeddings)


def answer_question(question: str, vectorstore: FAISS, chat_model, top_k: int) -> str:
    if not question.strip():
        raise ValueError("質問が空です。--question で質問文を指定してください。")

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return "関連する情報が見つかりませんでした。"

    context_blocks = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        file_name = doc.metadata.get("file_name", "不明ファイル")
        page_num = _format_page_number(doc.metadata)
        context_blocks.append(
            f"[資料{idx}] ファイル: {file_name}, ページ: {page_num}\n{doc.page_content}"
        )

    prompt = (
        "あなたはPDF要約アシスタントです。"
        "以下のコンテキストだけを根拠に日本語で回答してください。"
        "回答末尾に、利用した根拠を『ファイル名:ページ番号』形式で列挙してください。\n\n"
        f"質問:\n{question}\n\n"
        f"コンテキスト:\n{'\n\n'.join(context_blocks)}"
    )

    response = chat_model.invoke(prompt)
    answer_text = response.content if isinstance(response.content, str) else str(response.content)

    source_lines = []
    seen = set()
    for doc in retrieved_docs:
        file_name = doc.metadata.get("file_name", "不明ファイル")
        page_num = _format_page_number(doc.metadata)
        source = f"{file_name}:{page_num}"
        if source not in seen:
            seen.add(source)
            source_lines.append(f"- {source}")

    return (
        f"{answer_text.strip()}\n\n"
        "参照ソース\n"
        f"{'\n'.join(source_lines)}"
    )
