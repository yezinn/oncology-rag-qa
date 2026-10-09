"""
우선순위 3(Corrective RAG) 컴포넌트 단독 검증용 진단 스크립트 (2026-10-09).

qa_chain.py의 answer_question() 전체 플로우로 corrective 경로를 테스트하려면
1차 Guard(reranker)가 먼저 실패해야 하는데, 실측해보니 "ssGSEA"/"미토콘드리아"/
"면역세포 침윤" 같은 방법론 키워드가 겹치는 질문은 질병이 완전히 달라도(예:
뇌전증 vs 골관절염) 코퍼스 안의 방법론이 비슷한 논문 때문에 1차 Guard를
통과해버리는 경우가 있다 — 즉 "같은 방법론, 다른 질병"은 reranker Guard가
못 거르는(혹은 걸러선 안 될 수도 있는, 애매한) 또 다른 경계 사례다.

그래서 이 스크립트는 classify_scope()/corrective_pubmed_retry()를 1차 Guard
결과와 무관하게 직접 호출해서, Corrective RAG 컴포넌트 자체가 올바르게
동작하는지(스코프 판단이 맞는지, PubMed 재검색이 실제로 원하는 논문을
찾아오는지)만 따로 검증한다. 참고로 1차 Guard의 실제 통과/차단 여부도 같이
출력해서, 전체 파이프라인에서 왜 corrective가 트리거됐는지/안 됐는지 설명이
되게 한다.

사용법:
    python check_corrective_rag.py
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from qa_chain import (
    GEN_MODEL,
    RERANK_THRESHOLD,
    classify_scope,
    corrective_pubmed_retry,
    get_cross_encoder,
    get_vectorstore,
    rerank_top1_score,
)

# epilepsy_biomarker: fetch_pubmed_holdout.py로 수집한 "같은 쿼리로 검색되지만
# 151~180위라 색인 안 된" 논문(PMID 42128965) 기반 질문 — 코퍼스엔 없지만
# PubMed 재검색으로는 찾아와야 하는 진짜 테스트 케이스.
TEST_QUESTIONS = {
    "epilepsy_biomarker (in-scope, 코퍼스엔 없음, PMID 42128965 기대)":
        "뇌전증(epilepsy)에서 ssGSEA 기반 면역세포 침윤 분석과 머신러닝을 통해 "
        "발굴된 미토콘드리아 기능장애 관련 후보 바이오마커는 무엇인가?",
    "diabetes (out-of-scope, 재검색 자체가 안 일어나야 함)":
        "제2형 당뇨병의 인슐린 저항성 기전은 무엇인가?",
    "subway (out-of-scope)":
        "서울시 지하철 요금 체계는 어떻게 구성되어 있나요?",
}


def main():
    llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)
    vectorstore = get_vectorstore()
    print("Reranker 모델 로딩 중...")
    cross_encoder = get_cross_encoder()

    for label, question in TEST_QUESTIONS.items():
        print(f"\n=== {label} ===")
        print(f"Q: {question}")

        score = rerank_top1_score(cross_encoder, vectorstore, question)
        passed = score is not None and score >= RERANK_THRESHOLD
        score_str = f"{score:.4f}" if score is not None else "None"
        print(f"1차 Guard(reranker) top1_score: {score_str} (threshold {RERANK_THRESHOLD}) "
              f"-> {'통과(코퍼스 내 근거로 바로 답변)' if passed else '차단(corrective 기회 있음)'}")

        in_scope, topic, pubmed_query = classify_scope(question, llm)
        print(f"classify_scope -> in_scope={in_scope}, topic={topic!r}, pubmed_query={pubmed_query!r}")

        if not in_scope:
            print("corrective_pubmed_retry 호출 안 함 (스코프 게이트에서 차단)")
            continue

        docs = corrective_pubmed_retry(question, llm, cross_encoder)
        if docs:
            print(f"corrective_pubmed_retry -> {len(docs)}건 찾음 (관련성 재검증 통과):")
            for d in docs:
                print(f"  PMID {d.metadata.get('pmid')}: {d.metadata.get('title')[:80]}")
        else:
            print("corrective_pubmed_retry -> None (PubMed 검색 결과 없음 또는 관련성 threshold 미달)")


if __name__ == "__main__":
    main()
