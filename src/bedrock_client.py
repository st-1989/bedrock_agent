from __future__ import annotations

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

from config import Settings


def create_embeddings(settings: Settings) -> BedrockEmbeddings:
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    return BedrockEmbeddings(
        model_id=settings.embedding_model_id,
        client=client,
        region_name=settings.aws_region,
    )


def create_chat_model(settings: Settings) -> ChatBedrockConverse:
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    return ChatBedrockConverse(
        model=settings.chat_model_id,
        client=client,
        region_name=settings.aws_region,
        temperature=0,
    )
