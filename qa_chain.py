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
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

PERSIST_DIR = "./chroma_db"
# 2026-10-08: all-MiniLM-L6-v2(영어 전용) -> paraphrase-multilingual-MiniLM-L12-v2(다국어)
# 교체. 한국어 질문 + 영어 전용 임베딩 모델 조합에서 cross-lingual 불일치로 정답 문서가
# dense 검색에 전혀 안 걸리는 문제가 실측으로 확인됨(build_index.py 상단 주석 참고).
# build_index.py와 반드시 동일한 모델명이어야 함 — 다르면 쿼리/문서 벡터 공간이 어긋남.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5
# similarity_search_with_relevance_score는 0~1 값을 반환해야 정상(클수록 유사)이지만,
# 임베딩을 정규화하지 않고 Chroma 기본(L2) 거리를 쓰면 음수가 나오는 버그가 있었음.
# get_vectorstore()에서 normalize_embeddings=True + cosine space로 고침 (build_index.py도 동일 설정).
#
# !! 2026-10-08 임베딩 모델 교체로 재튜닝 필요 !!
# 아래 0.30은 all-MiniLM-L6-v2 기준으로 튜닝된 값으로, 모델이 바뀌면 점수 분포 자체가
# 달라져서 더 이상 유효하지 않음. build_index.py로 (기존 ./chroma_db 삭제 후) 재인덱싱한
# 다음, check_scores.py를 다시 돌려서 새 hit/miss 점수 분포를 보고 이 값을 다시 정할 것.
SCORE_THRESHOLD = 0.30  # TODO: check_scores.py로 재튜닝 전까지의 임시값

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


def bm25_preprocess(text):
    """BM25Retriever용 토크나이저 (기본 str.split() 대체).

    문제: 질문이 한국어 문장 안에 영문 전문용어가 조사 없이 그대로 붙는 경우가
    많음(예: "TNBC에서", "폐선암(LUAD)이", "ssGSEA와"). 한국어는 조사가 단어에
    공백 없이 붙기 때문에 기본 공백 분리로는 TNBC/LUAD/ssGSEA 같은 핵심 영문
    토큰이 한국어 조사·괄호와 뒤섞인 채로만 추출되어 영문 코퍼스와 전혀 매칭되지
    않음 (2026-10-08, debug_hybrid.py로 실측: 서로 다른 두 질문의 BM25 top5가
    완전히 동일하게 나와 점수가 전부 0으로 동률임을 확인).
    해결: 정규식으로 영문/숫자 알파뉴메릭 토큰만 소문자로 추출. 코퍼스(abstracts.json)
    쪽도 영어 텍스트라 같은 함수를 적용해도 정상적으로 단어 단위 토큰이 나옴 —
    Kiwi 같은 한국어 형태소 분석기를 추가로 들일 필요 없이 이 한 줄로 해결됨.
    """
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def get_hybrid_retriever(vectorstore):
    """Sparse(BM25) + Dense(Chroma) 하이브리드 리트리버 (RRF로 융합).

    주의: Grounding Guard의 통과/차단 판단은 answer_question()에서 여전히
    기존 dense cosine score(SCORE_THRESHOLD) 기준으로만 이뤄진다 — 그대로 유지.
    EnsembleRetriever의 RRF 융합 결과는 원래의 유사도 점수를 보존하지 않으므로
    Guard 판단 기준으로 쓸 수 없기 때문. 이 하이브리드 리트리버는 Guard를 통과한
    뒤 LLM에 전달할 컨텍스트 문서를 고르는 데에만 사용한다.

    BM25Retriever의 코퍼스는 build_index.py의 load_documents()를 그대로 재사용해
    data/abstracts.json에서 직접 구성한다 — Chroma 인덱스를 만들 때 쓴 것과 완전히
    동일한 소스/구성 코드라서 두 리트리버가 서로 다른 문서 집합을 보는(드리프트) 일이
    구조적으로 생기지 않는다.
    """
    from build_index import load_documents

    docs = load_documents()
    bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=bm25_preprocess)
    bm25_retriever.k = TOP_K

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5],
    )


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


def format_context_docs(docs):
    """하이브리드 리트리버가 반환하는 (score 없는) 순수 Document 리스트용 포맷터."""
    parts = []
    for doc in docs:
        pmid = doc.metadata.get("pmid", "unknown")
        parts.append(f"(PMID: {pmid}) {doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question, llm, vectorstore, hybrid_retriever=None):
    results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)

    # --- Grounding Guard: 근거가 불충분하면 LLM 호출 자체를 하지 않음 ---
    # (hybrid_retriever를 넘겨도 이 판단은 항상 dense cosine score 기준 그대로 — 변경 없음)
    if not results or results[0][1] < SCORE_THRESHOLD:
        return {
            "answer": "관련 근거 논문을 찾을 수 없어 확인해드릴 수 없습니다.",
            "sources": [],
            "grounded": False,
        }

    if hybrid_retriever is not None:
        # Guard를 통과한 뒤에는 BM25+Dense 하이브리드 검색 결과로 LLM 컨텍스트를 구성
        hybrid_docs = hybrid_retriever.invoke(question)
        context = format_context_docs(hybrid_docs)
        sources = [doc.metadata.get("pmid") for doc in hybrid_docs]
    else:
        context = format_context(results)
        sources = [doc.metadata.get("pmid") for doc, _ in results]

    chain = PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": extract_text(response),
        "sources": sources,
        "grounded": True,
    }


def main():
    llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)
    vectorstore = get_vectorstore()
    hybrid_retriever = get_hybrid_retriever(vectorstore)

    print("종료하려면 'exit' 입력")
    print("(Grounding Guard 판단: dense cosine score 기준 / LLM 컨텍스트: BM25+Dense 하이브리드)\n")
    while True:
        question = input("질문: ")
        if question.strip().lower() == "exit":
            break
        result = answer_question(question, llm, vectorstore, hybrid_retriever)
        print("\n답변:", result["answer"])
        print("출처 PMID:", result["sources"], "\n")


if __name__ == "__main__":
    main()
