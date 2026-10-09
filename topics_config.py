"""
pubmed-rag-qa가 다루는 연구 주제 정의 — 단일 소스(single source of truth).

이 프로젝트의 실제 사용 흐름은 "주제를 정한다 -> 그 주제의 논문을 모은다 ->
그 코퍼스 안에서 질문한다"이다. 지금까지는 그 "주제"가 세 군데(fetch_pubmed.py의
QUERIES, fetch_pubmed_holdout.py의 QUERIES, qa_chain.py의 CORRECTIVE_TOPICS)에
각각 하드코딩돼 있어서, 주제를 바꾸려면 세 파일을 일일이 맞춰 고쳐야 했다.
이 파일 하나로 모아서, 다른 주제로 바꾸고 싶을 때 TOPICS만 수정하면 된다.

각 값(쿼리 문자열)은 두 가지로 쓰인다:
    1) fetch_pubmed.py / fetch_pubmed_holdout.py: PubMed 검색 쿼리 그 자체
    2) qa_chain.py: Corrective RAG 스코프 게이트에서 "이 질문이 우리 도메인에
       속하는가"를 LLM이 판단할 때 참고하는 주제 설명

주제를 바꾼 뒤에는 순서대로 다시 실행해야 전체 파이프라인이 일관되게 바뀐다:
    1. python fetch_pubmed.py                      (새 주제로 초록 재수집)
    2. rm -rf chroma_db && python build_index.py   (반드시 재인덱싱 — 기존 벡터와
                                                     호환 안 됨)
    3. (선택) golden_set.json도 새 주제에 맞는 질문/정답 PMID로 다시 구성
    4. (선택) python fetch_pubmed_holdout.py        (경계선 예시 재료가 필요할 때만)

주제를 "추가"만 하고 싶을 때(기존 주제는 유지) 주의할 점: 새로 추가한 주제는
위 1~2번을 다시 실행해 코퍼스에 포함시키기 전까지는 Corrective RAG(실시간 PubMed
재검색)로만 응답한다 — 매 질문마다 NCBI API를 호출하므로 느리고, 사전 색인된
코퍼스만큼 결과가 안정적이지 않을 수 있다.
"""

TOPICS = {
    "egfr_luad_prognosis": "EGFR mutation lung adenocarcinoma prognosis biomarker",
    "tnbc_chemo_response": "triple negative breast cancer neoadjuvant chemotherapy response prediction",
    "ssgsea_pathway": "ssGSEA pathway enrichment gene expression",
    "transfer_learning_drug_response": "transfer learning cell line patient drug response prediction",
}
