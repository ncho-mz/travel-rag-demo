import os

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def _configure_bedrock_api_key() -> None:
    """Bridge local env var to Bedrock bearer token env var."""
    api_key = os.getenv("BEDROCK_API_KEY", "").strip()
    if api_key:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key


def _bedrock_client():
    _configure_bedrock_api_key()
    region = os.getenv("AWS_REGION", "us-east-1")
    profile = os.getenv("AWS_PROFILE")
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
    else:
        session = boto3.Session(region_name=region)
    return session.client("bedrock-runtime", region_name=region)


def get_embeddings():
    if _provider() == "bedrock":
        embedding_model = os.getenv(
            "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
        )
        return BedrockEmbeddings(model_id=embedding_model, client=_bedrock_client())
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_chat_model():
    if _provider() == "bedrock":
        model_id = os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        )
        return ChatBedrockConverse(model=model_id, temperature=0.2, client=_bedrock_client())
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0.2)
