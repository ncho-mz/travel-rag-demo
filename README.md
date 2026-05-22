# Travel RAG Demo

여행 코스 추천 챗봇 (FastAPI + Gradio + LangChain + FAISS)

## 1) 환경 준비

```bash
cd C:\Users\MZC01-NAKWON.CHO\Downloads\travel-rag-demo
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 환경 변수 설정

```bash
copy .env.example .env
```

`.env` 파일에서 `OPENAI_API_KEY` 값을 채워주세요.

### OpenAI를 사용할 때

- `LLM_PROVIDER=openai`
- `OPENAI_API_KEY` 설정

### AWS Bedrock을 사용할 때

- `LLM_PROVIDER=bedrock`
- `AWS_REGION` 설정 (예: `us-east-1`)
- 필요하면 `AWS_PROFILE` 설정
- API Key 방식이면 `BEDROCK_API_KEY` 설정
- `BEDROCK_MODEL_ID`, `BEDROCK_EMBEDDING_MODEL_ID` 설정

AWS 자격증명은 아래 중 하나로 준비해야 합니다.

- `aws configure`로 로컬 프로파일 구성
- 또는 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` 환경변수 설정

또한 Bedrock 콘솔에서 사용하려는 모델(예: Claude, Titan Embedding)에 대한 Model access가 열려 있어야 합니다.

#### Bedrock API Key 방식 (권장: 데모/테스트)

`.env` 예시:

```bash
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_API_KEY=your_bedrock_api_key_here
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
```

참고:

- Bedrock API Key는 리전 종속이므로 `AWS_REGION`과 키 생성 리전을 맞춰야 합니다.
- 이전에 OpenAI 임베딩으로 인덱싱한 경우, Bedrock으로 바꾼 뒤 `python scripts\ingest.py`를 다시 실행해야 합니다.

## 3) 문서 인덱싱

```bash
python scripts\ingest.py
```

## 4) API 실행

```bash
uvicorn app.main:app --reload --port 8000
```

## 5) UI 실행 (새 터미널)

```bash
python ui\gradio_app.py
```

브라우저에서 `http://127.0.0.1:7860` 접속

## 6) 질문 예시

- "제주 2박 3일 일정 추천해줘"
- "부산 비 오는 날 실내 코스 중심으로 짜줘"
- "도쿄 3박 4일, 첫날은 가볍게 걷는 코스로 구성해줘"

## Hugging Face Spaces 업로드

처음에는 로컬에서 먼저 동작 확인 후, Space에 올리는 것을 권장합니다.
Space는 `Gradio` SDK로 새로 만들고, 이 프로젝트 기준으로는 최소한 아래 파일이 필요합니다.

- `ui/gradio_app.py` (Space에서는 보통 파일명을 `app.py`로 둠)
- `requirements.txt`
- `app/`, `scripts/`, `data/` 폴더
