import gradio as gr
import requests


API_URL = "http://127.0.0.1:8000/chat"


def chat_fn(message, history):
    history = history or []
    response = requests.post(API_URL, json={"message": message}, timeout=90)
    response.raise_for_status()
    data = response.json()

    source_names = [src["source"] for src in data.get("sources", [])]
    source_text = ", ".join(source_names) if source_names else "없음"
    answer = f"{data['answer']}\n\n(출처: {source_text} / {data['latency_ms']}ms)"
    return answer


demo = gr.ChatInterface(
    fn=chat_fn,
    title="여행 코스 추천봇 (RAG 데모)",
    description="질문하면 문서 기반으로 여행 코스를 추천합니다.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
