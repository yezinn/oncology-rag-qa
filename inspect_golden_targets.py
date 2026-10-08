"""
Golden set의 '정답 문서'가 실제로 코퍼스에 어떻게 저장돼 있는지, 그리고 그 문서에 대한
질문의 dense 유사도 순위/점수가 실제로 얼마나 나오는지 직접 확인하는 진단 스크립트.

배경: BM25 토큰화 버그(bm25_preprocess)를 고쳤는데도 q-005/q-008의 BM25 top5가
수정 전과 완전히 동일하게 나왔고, Dense 단독도 같은 두 케이스에서 여전히 실패함.
BM25/Dense 둘 다 개별적으로 실패한다면 "의미 유사도는 낮지만 키워드는 일치"라는
원래 가설 자체가 이 3건에는 맞지 않는다는 뜻 — 코퍼스에 저장된 초록 내용이 질문과
실제로 얼마나 가까운지/정답 문서가 dense 순위에서 몇 위인지를 직접 봐야 판단 가능.

사용법:
    python inspect_golden_targets.py
"""
import json
from qa_chain import get_vectorstore

GOLDEN_SET_PATH = "golden_set.json"
DATA_PATH = "data/abstracts.json"
TARGET_IDS = {"q-001", "q-005", "q-008"}
SEARCH_DEPTH = 50  # top-5 밖에 정답이 있는지 보려고 더 깊게 검색


def main():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = {c["id"]: c for c in json.load(f)}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        abstracts = {a["pmid"]: a for a in json.load(f)}

    vectorstore = get_vectorstore()

    for case_id in sorted(TARGET_IDS):
        case = golden_set[case_id]
        expected_pmid = case["expected_pmids"][0]
        doc = abstracts.get(expected_pmid)

        print(f"=== [{case_id}] ===")
        print(f"질문: {case['question']}")

        if doc is None:
            print(f"!! 기대 PMID {expected_pmid}가 로컬 data/abstracts.json에 없음 (코퍼스 밖!)")
            print()
            continue

        print(f"정답 PMID {expected_pmid} 제목: {doc['title']}")
        print(f"초록(앞 300자): {doc['abstract'][:300]}...")

        # similarity_search_with_relevance_scores(k=5)는 top-5만 주므로, 정답 문서가
        # top-5 밖으로 밀려나 있으면 거기엔 안 나타남 -> 더 깊게(top-50) 조회해서
        # 정답 문서가 실제로 몇 위인지, 점수가 얼마인지 직접 확인
        results = vectorstore.similarity_search_with_relevance_scores(case["question"], k=SEARCH_DEPTH)
        rank, score = None, None
        for i, (d, s) in enumerate(results):
            if d.metadata.get("pmid") == expected_pmid:
                rank, score = i + 1, s
                break

        if rank is None:
            print(f"-> dense 검색 top-{SEARCH_DEPTH} 안에도 정답 문서가 없음 (그보다 더 밑)")
        else:
            print(f"-> dense 검색에서 정답 문서 순위: {rank}위 / score={score:.4f}")
        print()


if __name__ == "__main__":
    main()
