from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.config import VECTORSTORE_DIR
from app.llm import get_embeddings


def load_vectorstore(index_path: Path | None = None) -> FAISS:
    path = index_path or VECTORSTORE_DIR
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{path}'. Run 'python scripts/ingest.py' first."
        )
    return FAISS.load_local(
        folder_path=str(path),
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def get_retriever(k: int = 4):
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})
