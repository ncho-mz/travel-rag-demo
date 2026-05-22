import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.rag_chain import run_rag
from app.schemas import ChatRequest, ChatResponse, SourceItem


app = FastAPI(title="Travel RAG Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    started_at = time.time()
    answer, docs, trace = run_rag(payload.message)
    latency_ms = int((time.time() - started_at) * 1000)

    sources = [
        SourceItem(
            source=doc.metadata.get("source", "unknown"),
            content=doc.page_content[:220],
        )
        for doc in docs
    ]
    return ChatResponse(
        answer=answer, sources=sources, latency_ms=latency_ms, agent_trace=trace
    )
