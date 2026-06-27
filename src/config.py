from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    aws_region: str
    chat_model_id: str
    embedding_model_id: str
    data_dir: Path
    output_file: Path
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4


def load_settings() -> Settings:
    load_dotenv()

    root_dir = Path(__file__).resolve().parent.parent

    aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    chat_model_id = os.getenv("BEDROCK_CHAT_MODEL_ID")
    embedding_model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID")

    if not aws_region:
        raise ValueError("AWSリージョンが未設定です。.env の AWS_REGION を設定してください。")
    if not chat_model_id:
        raise ValueError(
            "ClaudeモデルIDが未設定です。.env の BEDROCK_CHAT_MODEL_ID を設定してください。"
        )
    if not embedding_model_id:
        raise ValueError(
            "EmbeddingモデルIDが未設定です。.env の BEDROCK_EMBEDDING_MODEL_ID を設定してください。"
        )

    return Settings(
        aws_region=aws_region,
        chat_model_id=chat_model_id,
        embedding_model_id=embedding_model_id,
        data_dir=root_dir / "data",
        output_file=root_dir / "outputs" / "answer.txt",
    )
