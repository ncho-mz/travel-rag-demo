from typing import List

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class SourceItem(BaseModel):
    source: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    latency_ms: int
    agent_trace: List[str]
