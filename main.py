"""
RAG QA를 웹 API로 제공하는 FastAPI 서버. qa_chain.py의 answer_question()을 그대로
재사용하며, 로직을 중복 구현하지 않는다. 정적 데모 UI(static/index.html)도 같은
서버에서 함께 서빙한다.

사전 준비:
    pip install fastapi uvicorn --break-system-packages
    export GOOGLE_API_KEY="..."
    (build_index.py로 만든 ./chroma_db 필요)

사용법:
    uvicorn main:app --reload
    -> http://localhost:8000        (데모 UI)
    -> http://localhost:8000/docs   (Swagger UI)
"""
from typing import List, Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from qa_chain import GEN_MODEL, SCORE_THRESHOLD, TOP_K, answer_question, get_vectorstore

app = FastAPI(title="PubMed RAG QA API")

_llm = None
_vectorstore = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)
    return _llm


def get_store():
    # 프로세스당 한 번만 로드 (요청마다 임베딩 모델을 다시 로드하지 않도록)
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore()
    return _vectorstore


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    grounded: bool
    top1_score: Optional[float]
    score_threshold: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/qa/ask", response_model=AskResponse)
def ask(req: AskRequest):
    vectorstore = get_store()

    # Grounding Guard의 판단 근거인 top1_score를 UI에 그대로 노출하기 위해 한 번 더 검색
    # (answer_question 내부에서도 같은 검색을 하지만, 점수를 API 응답으로 반환하려면 필요)
    results = vectorstore.similarity_search_with_relevance_scores(req.question, k=TOP_K)
    top1_score = results[0][1] if results else None

    result = answer_question(req.question, get_llm(), vectorstore)

    return AskResponse(
        answer=result["answer"],
        sources=[str(s) for s in result["sources"]],
        grounded=result["grounded"],
        top1_score=top1_score,
        score_threshold=SCORE_THRESHOLD,
    )


# 반드시 모든 API 라우트 정의 이후(맨 마지막)에 와야 /health, /docs 같은 기존 경로를 안 가림
app.mount("/", StaticFiles(directory="static", html=True), name="static")
