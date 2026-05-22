from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model
from app.retriever import get_retriever


PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful travel course planner assistant.
Answer in Korean.
Use only the context provided below.
If the context is not enough, say "주어진 자료에서 충분한 정보를 찾지 못했어요."
Keep answers practical with day-by-day suggestions when relevant.

Context:
{context}

Question:
{question}
"""
)


def _join_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def run_rag(question: str) -> tuple[str, list[Document]]:
    retriever = get_retriever(k=4)
    docs = retriever.invoke(question)
    context = _join_docs(docs)

    chain = PROMPT | get_chat_model()
    answer = chain.invoke({"context": context, "question": question}).content
    return answer, docs
