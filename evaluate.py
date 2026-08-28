"""
Golden Set 기반 평가 스크립트.
retrieval hit rate (기대 PMID가 실제로 검색되었는지)를 자동으로 측정합니다.
(SKALA HelpDesk AI 프로젝트의 Golden Set 평가 방식과 동일한 접근)

사전 준비:
    golden_set_template.json을 복사해 golden_set.json으로 만들고, 본인 지식으로
    질문과 정답 PMID를 채워넣으세요 (최소 10~15개 권장).

사용법:
    python evaluate.py
"""
import json
from qa_chain import get_vectorstore, TOP_K

GOLDEN_SET_PATH = "golden_set.json"


def main():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    vectorstore = get_vectorstore()

    hits = 0
    results_log = []
    for case in golden_set:
        docs = vectorstore.similarity_search(case["question"], k=TOP_K)
        retrieved_pmids = [d.metadata.get("pmid") for d in docs]
        expected = set(case["expected_pmids"])
        hit = bool(expected & set(retrieved_pmids))
        hits += int(hit)
        results_log.append({
            "id": case["id"],
            "question": case["question"],
            "expected_pmids": case["expected_pmids"],
            "retrieved_pmids": retrieved_pmids,
            "hit": hit,
        })

    hit_rate = hits / len(golden_set) if golden_set else 0
    print(f"Retrieval Hit Rate: {hit_rate:.2f} ({hits}/{len(golden_set)})")

    with open("evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump({"hit_rate": hit_rate, "cases": results_log}, f, ensure_ascii=False, indent=2)
    print("상세 결과 -> evaluation_result.json")


if __name__ == "__main__":
    main()
