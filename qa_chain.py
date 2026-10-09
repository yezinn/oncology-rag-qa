"""
RAG 기반 QA 체인.
검색된 근거(초록)의 유사도가 임계값보다 낮으면 LLM을 호출하지 않고 즉시
"확인 불가"로 응답하는 Grounding Guard를 코드 레벨에서 강제한다. Guard가
1차로 차단해도, 질문이 우리가 다루는 도메인(CORRECTIVE_TOPICS)에 속한다고
판단되면 1회 한정으로 실시간 PubMed 재검색을 시도하는 Corrective RAG 단계가
있다(corrective_pubmed_retry() 참고) — 정적 코퍼스(598건)에 없을 뿐 실제로는
답할 수 있는 질문을 구제하기 위함.

사전 준비:
    pip install langchain-google-genai biopython --break-system-packages
    export GOOGLE_API_KEY="..."
    export ENTREZ_EMAIL="..."

사용법:
    python qa_chain.py
"""
import os
import re
from Bio import Entrez
from topics_config import TOPICS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
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
# 2026-10-08 임베딩 모델 교체(paraphrase-multilingual-MiniLM-L12-v2)에 맞춰 재튜닝함.
# check_scores.py(golden set 14개, in-domain) + check_offtopic_scores.py(코퍼스 밖
# 질문)로 실측한 점수 분포:
#   - golden set 14개 in-domain 질문의 top1_score 최소값: 0.4919
#   - 완전히 무관한 질문(지하철 요금/저녁 메뉴/월드컵) top1_score: 0.1595~0.1851
#   - "의학적으로 그럴듯하지만 코퍼스 밖 도메인"인 adversarial 질문(제2형 당뇨병
#     인슐린 저항성 — 이 코퍼스는 암 유전체 전문): top1_score 0.7098
# 즉 "완전히 무관한 주제"는 0.19 이하로 깔끔하게 분리되지만, "생물의학적으로 그럴듯한
# 오답"은 오히려 진짜 정답(0.49)보다도 높게 나옴 — top-1 cosine 임계값 하나로는
# 후자를 걸러낼 수 없음(구조적 한계, 단일 threshold로 해결 불가). SCORE_THRESHOLD는
# 더 이상 Guard 판단에 쓰지 않지만(아래 RERANK_THRESHOLD 참고), get_reranker_retriever가
# 없는 경로의 호환성을 위해 상수는 남겨둔다.
SCORE_THRESHOLD = 0.40

# 우선순위 2 (Reranker, 2026-10-08). 다국어 지원 모델을 반드시 써야 함 — 영어 전용
# bge-reranker-base를 썼다면 한국어 질문에서 §5와 똑같은 cross-lingual 문제를
# reranker 단계에서 반복했을 것. bge-reranker-v2-m3는 다국어 지원.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# reranker에 넘길 dense 후보 풀 크기. 최종적으로는 top_n(=TOP_K)개로 줄이지만,
# 후보 풀 자체가 너무 좁으면(k=TOP_K) 애초에 정답이 후보에 없어서 reranker가
# 재정렬할 것이 없음 — §5에서 Reranker가 "1차 검색이 이미 추려온 후보 안에서만
# 재정렬한다"는 한계를 짚었던 것과 같은 이유로 더 넓게 가져옴.
RERANK_CANDIDATE_K = 15
# check_reranker_scores.py 실측(2026-10-08, golden set 14개 + off-topic 3개 +
# adversarial 1개): in-domain 최소 0.9629 / off-topic 최대 0.0049 / adversarial
# (제2형 당뇨병 인슐린 저항성) 최대 0.0709. dense cosine에서는 이 adversarial
# 케이스가 in-domain 최소(0.4919)보다도 높게(0.7098) 나와 구분이 안 됐지만,
# Cross-Encoder는 질문+문서를 함께 평가(joint encoding)하기 때문에 "생물의학
# 용어/어투는 비슷해도 실제 내용은 다른 도메인"이라는 세부 불일치를 훨씬 잘
# 잡아냄 — in-domain과 adversarial 사이에 약 13배(0.96 vs 0.07) 마진이 생김.
# 그래서 Grounding Guard 판단 기준을 dense cosine(SCORE_THRESHOLD)에서 이
# reranker score로 교체한다. 0.5는 그 큰 마진 사이 어디든 잡아도 안전하지만,
# 임의로 중간값을 쓰기보다 실측값에 안전 마진을 둔 값.
RERANK_THRESHOLD = 0.5

# 우선순위 3 (Corrective RAG, 2026-10-09). Guard가 차단했을 때 바로 "근거 부족"을
# 반환하는 대신, 1회 한정으로 실시간 PubMed 검색을 시도한다(fetch_pubmed.py의
# Entrez 자산 재사용).
#
# 주의: 이건 RERANK_THRESHOLD(관련성 — "이 문서가 질문과 관련 있는가")와는
# 완전히 다른 축의 판단이 하나 더 필요하다. 재검색을 허용할지를 관련성 점수로만
# 판단하면 안 됨 — PubMed 전체에는 당뇨병 인슐린 저항성에 관한 "진짜로 관련
# 있는" 논문이 실제로 존재하고, reranker는 그 논문을 찾으면 당연히 높은 점수를
# 줄 것이기 때문. 즉 관련성만 보면 §6에서 막은 "그럴듯한 오답"이 재검색을 통해
# 뒷문으로 다시 통과해버림. 그래서 재검색 자체를 허용할지는 "이 질문이 우리가
# 다루기로 선언한 4개 주제 중 하나에 속하는가"라는 별도의 스코프 게이트로
# 판단해야 한다(classify_scope() 참고) — fetch_pubmed.py가 코퍼스를 수집할 때
# 썼던 주제 쿼리를 "우리 도메인의 정의"로 그대로 재사용. 주의: 이 4개 주제가
# 전부 "암"으로 제한된 건 아님(예: ssgsea_pathway는 방법론 기준이라 코퍼스에도
# 골관절염·치주염·파킨슨병 등 암이 아닌 질병이 이미 섞여 있음) — classify_scope()
# 프롬프트에 "암이어야 한다"는 조건을 넣으면 안 되는 이유이기도 함(2026-10-09,
# 뇌전증 테스트 질문으로 이 실수를 실측으로 확인·수정함).
#
# 2026-10-09: 주제 목록 자체는 topics_config.py의 TOPICS 하나로 모아뒀다 —
# fetch_pubmed.py(코퍼스 수집)와 여기(스코프 게이트)가 서로 다른 목록을 보면
# "코퍼스에는 있는데 스코프 게이트는 막는다"거나 그 반대의 불일치가 생기기
# 때문. 다루는 주제를 바꾸고 싶으면 이 파일이 아니라 topics_config.py를 고치면
# 된다(그 파일 docstring에 전체 절차 설명).
ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "")
CORRECTIVE_TOPICS = TOPICS
# NCBI E-utilities rate limit(API 키 없이는 초당 3회)을 고려해 재검색 1회당 적게 가져옴.
CORRECTIVE_MAX_RESULTS = 5

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

    LLM에 전달할 컨텍스트 문서를 고르는 데에만 사용한다(Guard 판단과는 분리 —
    아래 answer_question()의 cross_encoder 설명 참고).

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


def get_reranker_retriever(vectorstore, top_n=TOP_K, candidate_k=RERANK_CANDIDATE_K):
    """Dense 후보를 Cross-Encoder로 재정렬하는 리트리버.

    주의: CrossEncoderReranker.compress_documents()는 재정렬된 Document만 돌려주고
    원래 rerank score는 메타데이터에 남기지 않는다(langchain_classic 소스 확인,
    2026-10-08) — score 자체가 필요하면(Guard 판단 등) 이 함수가 아니라
    rerank_top1_score()처럼 HuggingFaceCrossEncoder(...).score(...)를 직접 호출해야 함.
    """
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": candidate_k})
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


def get_cross_encoder():
    """Guard 판단용 Cross-Encoder 모델을 한 번만 로드해서 재사용하기 위한 헬퍼."""
    return HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)


def rerank_top1_score(cross_encoder, vectorstore, question, candidate_k=RERANK_CANDIDATE_K):
    """Guard 판단에 쓸 reranker top1 score만 계산한다.

    get_reranker_retriever()는 재정렬된 Document 리스트만 반환하고 점수는 버리므로
    (위 주석 참고) Guard처럼 "점수 자체"가 필요한 곳에서는 이 함수를 쓴다.
    check_reranker_scores.py의 top1_rerank()와 동일한 로직(후보는 dense만 사용).
    """
    candidates = vectorstore.similarity_search(question, k=candidate_k)
    if not candidates:
        return None
    pairs = [(question, d.page_content) for d in candidates]
    scores = list(cross_encoder.score(pairs))
    return max(scores)


def classify_scope(question, llm):
    """질문이 CORRECTIVE_TOPICS 중 하나에 명확히 속하는지 LLM으로 판단하고,
    속한다면 실시간 PubMed 검색에 쓸 영어 쿼리를 함께 생성한다.

    주의: 이건 "문서가 질문과 관련 있는가"(rerank_top1_score, 관련성)와는 다른
    질문이다 — "이 질문이 애초에 우리가 다루기로 선언한 도메인에 속하는가"
    (스코프)를 본다. 이 둘을 섞으면 안 되는 이유는 CORRECTIVE_TOPICS 상단 주석
    참고. 실패(API 오류 등)하면 안전한 쪽으로 fallback(in_scope=False, 재검색 없음).
    """
    from pydantic import BaseModel, Field

    class ScopeResult(BaseModel):
        in_scope: bool = Field(description="질문이 아래 4개 주제 중 하나에 명확히 속하는가")
        topic: str = Field(description="속한다면 topic key, 아니면 빈 문자열")
        pubmed_query: str = Field(description="속한다면 PubMed 검색에 쓸 영어 키워드 쿼리(5~8단어), 아니면 빈 문자열")

    topics_desc = "\n".join(f"- {k}: {v}" for k, v in CORRECTIVE_TOPICS.items())
    prompt = f"""다음은 이 시스템이 실제로 다루는 연구 주제들의 설명입니다. 질문의
내용(방법론·연구대상)이 이 설명 중 하나와 실질적으로 일치하는지만 판단하세요.

중요: 각 주제는 설명에 적힌 그대로의 범위입니다 — 특정 질병명이 설명에 없다면
그 주제는 질병으로 제한된 것이 아니라 방법론(예: ssGSEA 기반 분석)으로 정의된
것이므로, 설명에 언급되지 않은 질병이라도 그 방법론을 묻는 질문이면 속하는
것으로 판단하세요. 임의로 "암 관련 주제여야 한다"처럼 설명에 없는 조건을
추가하지 마세요.

설명 중 어느 것과도 방법론·연구대상이 실질적으로 일치하지 않는 질문(예: 당뇨병
인슐린 저항성 기전처럼 이 주제들과 무관한 생물의학 주제, 지하철 요금 체계처럼
비의학적 주제)이라면 반드시 in_scope=False로 답하세요.

[주제 목록]
{topics_desc}

[질문]
{question}
"""
    try:
        structured_llm = llm.with_structured_output(ScopeResult)
        result = structured_llm.invoke(prompt)
        return result.in_scope, result.topic, result.pubmed_query
    except Exception:  # noqa: BLE001 - 분류 실패 시 안전하게 재검색 없이 차단
        return False, "", ""


def corrective_pubmed_retry(question, llm, cross_encoder, top_n=TOP_K):
    """Guard가 1차로 차단했을 때 1회 한정으로 시도하는 Corrective RAG 재검색.

    classify_scope()가 "우리 도메인에 속한다"고 판단한 경우에만 실시간 PubMed
    검색을 수행하고(fetch_pubmed.py의 Entrez 패턴 재사용), 가져온 문서도
    rerank_top1_score()와 동일한 RERANK_THRESHOLD로 다시 검증한다 — 스코프 게이트와
    관련성 검증을 모두 통과해야만 컨텍스트로 쓴다. 코퍼스에 영구히 추가하지는
    않는다(이번 질문에 한정된 임시 보강).
    """
    in_scope, topic, pubmed_query = classify_scope(question, llm)
    if not in_scope or not pubmed_query:
        return None

    Entrez.email = ENTREZ_EMAIL or "unknown@example.com"
    try:
        handle = Entrez.esearch(db="pubmed", term=pubmed_query, retmax=CORRECTIVE_MAX_RESULTS, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        pmids = record.get("IdList", [])
        if not pmids:
            return None
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
    except Exception:  # noqa: BLE001 - NCBI 호출 실패 시 재검색 없이 기존 차단으로 fallback
        return None

    from langchain_core.documents import Document

    docs = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            pmid = str(medline["PMID"])
            article_data = medline["Article"]
            title = str(article_data.get("ArticleTitle", ""))
            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts)
            if abstract:
                content = f"Title: {title}\n\nAbstract: {abstract}"
                docs.append(Document(page_content=content, metadata={"pmid": pmid, "title": title, "topic": topic}))
        except (KeyError, IndexError):
            continue
    if not docs:
        return None

    pairs = [(question, d.page_content) for d in docs]
    scores = list(cross_encoder.score(pairs))
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    if ranked[0][1] < RERANK_THRESHOLD:
        return None
    return [d for d, _ in ranked[:top_n]]


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


def answer_question(question, llm, vectorstore, hybrid_retriever=None, cross_encoder=None):
    # --- Grounding Guard: 근거가 불충분하면 LLM 호출 자체를 하지 않음 ---
    # cross_encoder가 주어지면 reranker score 기준(RERANK_THRESHOLD)으로 판단하고,
    # 주어지지 않으면 기존 dense cosine score 기준(SCORE_THRESHOLD)으로 판단한다
    # (하위 호환 — cross_encoder 없이 호출하던 기존 코드는 그대로 동작).
    # 2026-10-08: dense cosine 단일 threshold는 "의학적으로 그럴듯하지만 코퍼스
    # 밖 도메인"인 adversarial 질문을 못 거르는 구조적 한계가 있음
    # (check_offtopic_scores.py: adversarial 0.7098 > in-domain 최소 0.4919).
    # check_reranker_scores.py로 같은 질문들을 reranker score로 실측하니 in-domain
    # 최소 0.9629 / adversarial 최대 0.0709로 명확히 분리됨 — 그래서 가능하면
    # cross_encoder 기준을 쓴다 (RERANKER_MODEL 상단 주석 참고).
    if cross_encoder is not None:
        score = rerank_top1_score(cross_encoder, vectorstore, question)
        grounded_ok = score is not None and score >= RERANK_THRESHOLD
    else:
        results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
        grounded_ok = bool(results) and results[0][1] >= SCORE_THRESHOLD

    if not grounded_ok:
        # --- 우선순위 3, Corrective RAG: 1차 차단 시 1회 한정 실시간 재검색 ---
        # cross_encoder가 있을 때만 시도(스코프 게이트·관련성 재검증 모두 reranker
        # 기준이 필요하므로). CORRECTIVE_TOPICS 상단 주석 참고 — 관련성이 아니라
        # 스코프(classify_scope)로 먼저 걸러야 당뇨병류 질문이 뒷문으로 통과하지 않음.
        if cross_encoder is not None:
            corrective_docs = corrective_pubmed_retry(question, llm, cross_encoder)
            if corrective_docs:
                context = format_context_docs(corrective_docs)
                sources = [doc.metadata.get("pmid") for doc in corrective_docs]
                chain = PROMPT | llm
                response = chain.invoke({"context": context, "question": question})
                return {
                    "answer": extract_text(response),
                    "sources": sources,
                    "grounded": True,
                    "corrective": True,
                    "context": context,
                }

        return {
            "answer": "관련 근거 논문을 찾을 수 없어 확인해드릴 수 없습니다.",
            "sources": [],
            "grounded": False,
            "corrective": False,
            "context": "",
        }

    if hybrid_retriever is not None:
        # Guard를 통과한 뒤에는 BM25+Dense 하이브리드 검색 결과로 LLM 컨텍스트를 구성
        hybrid_docs = hybrid_retriever.invoke(question)
        context = format_context_docs(hybrid_docs)
        sources = [doc.metadata.get("pmid") for doc in hybrid_docs]
    else:
        results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
        context = format_context(results)
        sources = [doc.metadata.get("pmid") for doc, _ in results]

    chain = PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": extract_text(response),
        "sources": sources,
        "grounded": True,
        "corrective": False,
        "context": context,
    }


def main():
    llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)
    vectorstore = get_vectorstore()
    hybrid_retriever = get_hybrid_retriever(vectorstore)
    print("Reranker 모델 로딩 중 (Grounding Guard 판단용)...")
    cross_encoder = get_cross_encoder()

    print("\n종료하려면 'exit' 입력")
    print("(Grounding Guard 판단: reranker score 기준 / LLM 컨텍스트: BM25+Dense 하이브리드")
    print(" / 1차 차단 시 스코프 게이트 통과하면 1회 한정 실시간 PubMed 재검색 시도)\n")
    while True:
        question = input("질문: ")
        if question.strip().lower() == "exit":
            break
        result = answer_question(question, llm, vectorstore, hybrid_retriever, cross_encoder)
        print("\n답변:", result["answer"])
        print("출처 PMID:", result["sources"])
        if result.get("corrective"):
            print("(실시간 PubMed 재검색으로 찾은 근거입니다 — 기존 정적 코퍼스에는 없던 문서)")
        print()


if __name__ == "__main__":
    main()
