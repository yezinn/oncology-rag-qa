"""
가설 검증용 진단 스크립트 (인덱스 재구축 없음, 기존 벡터스토어 그대로 사용):
"한국어 질문 + 영어 전용 임베딩 모델(all-MiniLM-L6-v2) 조합이 cross-lingual
의미 정렬을 제대로 못 해서 dense 검색이 실패하는 것"이라는 가설을,
같은 질문을 영어로 번역해 같은 벡터스토어에 질의해보는 것으로 싸게 검증한다.
(재인덱싱 없이 "쿼리만 영어로 바꿔서" 테스트하는 것이라 몇 초 안에 끝남)

사용법:
    python test_translated_query.py
"""
from qa_chain import get_vectorstore

SEARCH_DEPTH = 50

# q-001/q-005/q-008의 영어 번역 (의미만 보존, 직역은 아님)
CASES = [
    {
        "id": "q-001",
        "question_ko": "EGFR 변이 폐선암(LUAD)이 야생형 대비 면역치료 반응이 낮은 이유를 단일세포 전사체 수준에서 분석했을 때, 종양미세환경(TME)에서 결핍되어 있던 핵심 세포는 무엇인가?",
        "question_en": "Using single-cell transcriptome analysis to explain why EGFR-mutant lung adenocarcinoma (LUAD) shows lower immunotherapy response than wild-type, what key cell type was depleted in the tumor microenvironment (TME)?",
        "expected_pmid": "35140113",
    },
    {
        "id": "q-005",
        "question_ko": "TNBC에서 면역관문억제제(ICB) 반응을 예측하는 공간적(spatial) 종양미세환경 특징으로 가장 우세했던 것은 무엇인가?",
        "question_en": "In triple-negative breast cancer (TNBC), what spatial tumor microenvironment feature was the strongest predictor of response to immune checkpoint blockade (ICB)?",
        "expected_pmid": "37674077",
    },
    {
        "id": "q-008",
        "question_ko": "골관절염(OA)에서 ssGSEA와 머신러닝 기반 다중오믹스 분석을 통해 도출된 핵심 미토콘드리아 관련 유전자는 몇 개이며 무엇인가?",
        "question_en": "In osteoarthritis (OA), how many key mitochondria-related genes were identified through ssGSEA and machine learning-based multi-omics analysis, and what are they?",
        "expected_pmid": "39026663",
    },
]


def find_rank(vectorstore, question, expected_pmid):
    results = vectorstore.similarity_search_with_relevance_scores(question, k=SEARCH_DEPTH)
    for i, (d, s) in enumerate(results):
        if d.metadata.get("pmid") == expected_pmid:
            return i + 1, s
    return None, None


def main():
    vectorstore = get_vectorstore()

    for case in CASES:
        ko_rank, ko_score = find_rank(vectorstore, case["question_ko"], case["expected_pmid"])
        en_rank, en_score = find_rank(vectorstore, case["question_en"], case["expected_pmid"])

        print(f"=== [{case['id']}] ===")
        ko_str = f"{ko_rank}위 (score={ko_score:.4f})" if ko_rank else f"top-{SEARCH_DEPTH} 밖"
        en_str = f"{en_rank}위 (score={en_score:.4f})" if en_rank else f"top-{SEARCH_DEPTH} 밖"
        print(f"  한국어 질문  -> {ko_str}")
        print(f"  영어 번역 질문 -> {en_str}")
        print()


if __name__ == "__main__":
    main()
