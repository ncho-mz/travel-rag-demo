from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model
from app.retriever import get_retriever


PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a travel planning triage agent.
Answer in Korean.
Given the user question, extract and organize planning requirements.
Create a compact planning brief with these sections:
1) 여행지/후보
2) 일정 길이
3) 여행 스타일(예: 미식, 자연, 쇼핑)
4) 제약/요청사항
5) 추천 일정 설계 포인트

Question:
{question}
"""
)


CITY_EXPERT_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful travel city expert agent.
Answer in Korean.
Use only the context provided below.
If the context is not enough, say "주어진 자료에서 충분한 정보를 찾지 못했어요."
Keep answers practical with day-by-day suggestions when relevant.

Planning brief:
{planning_brief}

Context:
{context}

Question:
{question}
"""
)


def _join_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def run_rag(question: str) -> tuple[str, list[Document], list[str]]:
    trace: list[str] = []
    model = get_chat_model()

    planner_chain = PLANNER_PROMPT | model
    planning_brief = planner_chain.invoke({"question": question}).content
    trace.append("planner_agent: 사용자 질문을 일정 요구사항으로 정리했습니다.")

    retriever = get_retriever(k=4)
    retrieval_query = f"{question}\n\n[planning_brief]\n{planning_brief}"
    docs = retriever.invoke(retrieval_query)
    context = _join_docs(docs)
    trace.append(f"city_expert_agent: 관련 문서 {len(docs)}개를 검색했습니다.")

    expert_chain = CITY_EXPERT_PROMPT | model
    answer = expert_chain.invoke(
        {"planning_brief": planning_brief, "context": context, "question": question}
    ).content
    trace.append("city_expert_agent: 최종 여행 코스를 생성했습니다.")
    return answer, docs, trace
