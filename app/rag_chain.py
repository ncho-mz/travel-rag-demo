from contextlib import nullcontext

try:
    from ddtrace import tracer
except Exception:  # pragma: no cover - ddtrace may not exist locally
    tracer = None

try:
    from ddtrace.llmobs import LLMObs
except Exception:  # pragma: no cover - ddtrace may not exist locally
    LLMObs = None

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


def _agent_span(span_name: str):
    if tracer is None:
        return nullcontext()
    return tracer.trace(span_name, service="travel-rag-demo-api", resource=span_name)


def _llmobs_task(name: str):
    if LLMObs is None:
        return nullcontext()
    return LLMObs.task(name=name)


def _llmobs_retrieval(name: str):
    if LLMObs is None:
        return nullcontext()
    retrieval_fn = getattr(LLMObs, "retrieval", None)
    if retrieval_fn is None:
        return nullcontext()
    return retrieval_fn(name=name)


def _llmobs_workflow(name: str):
    if LLMObs is None:
        return nullcontext()
    return LLMObs.workflow(name=name)


def run_rag(question: str) -> tuple[str, list[Document], list[str]]:
    trace: list[str] = []
    model = get_chat_model()

    with _llmobs_workflow("travel_rag_workflow") as workflow_span:
        planner_chain = (PLANNER_PROMPT | model).with_config({"run_name": "planner_agent"})
        with _llmobs_task("planner_agent"), _agent_span("agent.planner") as planner_span:
            if planner_span is not None:
                planner_span.set_tag("agent.name", "planner")
            planning_brief = planner_chain.invoke({"question": question}).content
            if LLMObs is not None:
                LLMObs.annotate(
                    input_data=question,
                    output_data=planning_brief,
                    tags={"agent": "planner"},
                )
        trace.append("planner_agent: 사용자 질문을 일정 요구사항으로 정리했습니다.")

        retriever = get_retriever(k=4)
        retrieval_query = f"{question}\n\n[planning_brief]\n{planning_brief}"
        with _llmobs_retrieval("city_retrieval"), _agent_span(
            "agent.retrieval"
        ) as retrieval_span:
            if retrieval_span is not None:
                retrieval_span.set_tag("agent.name", "retrieval")
            docs = retriever.invoke(retrieval_query)
            if LLMObs is not None:
                retrieval_output = [
                    {
                        "text": doc.page_content[:500],
                        "name": doc.metadata.get("source", "unknown"),
                        "id": str(i),
                    }
                    for i, doc in enumerate(docs)
                ]
                LLMObs.annotate(
                    input_data=retrieval_query,
                    output_data=retrieval_output,
                    tags={"agent": "retrieval"},
                )
        context = _join_docs(docs)
        trace.append(f"city_expert_agent: 관련 문서 {len(docs)}개를 검색했습니다.")

        expert_chain = (CITY_EXPERT_PROMPT | model).with_config(
            {"run_name": "city_expert_agent"}
        )
        with _llmobs_task("city_expert_agent"), _agent_span(
            "agent.city_expert"
        ) as expert_span:
            if expert_span is not None:
                expert_span.set_tag("agent.name", "city_expert")
            answer = expert_chain.invoke(
                {"planning_brief": planning_brief, "context": context, "question": question}
            ).content
            if LLMObs is not None:
                LLMObs.annotate(
                    input_data=question,
                    output_data=answer,
                    tags={"agent": "city_expert"},
                )
        if LLMObs is not None and workflow_span is not None:
            LLMObs.annotate(
                span=workflow_span,
                input_data=question,
                output_data=answer,
                tags={"agent": "workflow"},
            )
        trace.append("city_expert_agent: 최종 여행 코스를 생성했습니다.")
        return answer, docs, trace
