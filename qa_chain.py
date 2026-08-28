"""
RAG 기반 QA 체인.
검색된 근거(초록)의 유사도가 임계값보다 낮으면 LLM을 호출하지 않고 즉시
"확인 불가"로 응답하는 Grounding Guard를 코드 레벨에서 강제한다.

사전 준비:
    pip install langchain-google-genai --break-system-packages
    export GOOGLE_API_KEY="..."
    
사용법:
    python qa_chain.py
"""
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
# similarity_search_with_relevance_score는 0~1 값을 반환해야 정상(클수록 유사)이지만,
# 임베딩을 정규화하지 않고 Chroma 기본(L2) 거리를 쓰면 음수가 나오는 버그가 있었음.
# get_vectorstore()에서 normalize_embeddings=True + cosine space로 고침 (build_index.py도 동일 설정).
# 아래 값은 check_scores.py로 실측한 hit/miss 점수 분포를 보고 정한 값입니다.
# (Golden Set 14문항 기준) threshold 0.35 -> 0.30: 오답(hit=False) 통과율은 1/3로 동일하게
# 유지하면서, 정답(hit=True) 통과율만 6/11 -> 7/11로 개선되어 0.30을 채택함.
SCORE_THRESHOLD = 0.30

# 무료 티어(Google AI Studio) 모델. gemini-2.5-flash-lite는 신규 사용자에게 더 이상
# 제공되지 않아(2026-08 기준) gemini-3.5-flash-lite로 교체함. 모델명이 또 바뀌었다면
# https://aistudio.google.com/ 에서 현재 사용 가능한 -flash-lite 계열 모델명을 확인하세요.
GEN_MODEL = "gemini-3.5-flash-lite"

PROMPT = ChatPromptTemplate.from_template(
    """당신은 암 유전체/전사체 연구 도메인 전문 어시스턴트입니다.
아래 제공된 논문 초록만을 근거로 질문에 답하세요. 초록에 없는 내용은 절대 답변에 포함하지 마세요.
각 주장 뒤에는 반드시 (PMID: xxxxx) 형태로 출처를 표기하세요.

[검색된 논문 초록]
{context}

[질문]
{question}

[답변]"""
)


def get_vectorstore():
    # build_index.py와 반드시 동일한 임베딩 설정(normalize_embeddings=True)을 써야
    # 쿼리 임베딩과 저장된 문서 임베딩의 스케일이 맞아 relevance score가 의미를 가짐.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def extract_text(response):
    """langchain-google-genai 최신 버전은 response.content가 순수 문자열이 아니라
    [{"type": "text", "text": ..., "extras": {"signature": ...}}] 형태의 리스트로
    올 수 있음(Gemini의 thought-signature 메타데이터 포함). 두 경우 모두 안전하게
    순수 텍스트만 추출한다."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def format_context(docs_with_scores):
    parts = []
    for doc, _score in docs_with_scores:
        pmid = doc.metadata.get("pmid", "unknown")
        parts.append(f"(PMID: {pmid}) {doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question, llm, vectorstore):
    results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)

    # --- Grounding Guard: 근거가 불충분하면 LLM 호출 자체를 하지 않음 ---
    if not results or results[0][1] < SCORE_THRESHOLD:
        return {
            "answer": "관련 근거 논문을 찾을 수 없어 확인해드릴 수 없습니다.",
            "sources": [],
            "grounded": False,
        }

    context = format_context(results)
    chain = PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": extract_text(response),
        "sources": [doc.metadata.get("pmid") for doc, _ in results],
        "grounded": True,
    }


def main():
    llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)
    vectorstore = get_vectorstore()

    print("종료하려면 'exit' 입력\n")
    while True:
        question = input("질문: ")
        if question.strip().lower() == "exit":
            break
        result = answer_question(question, llm, vectorstore)
        print("\n답변:", result["answer"])
        print("출처 PMID:", result["sources"], "\n")


if __name__ == "__main__":
    main()
