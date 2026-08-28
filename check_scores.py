"""
SCORE_THRESHOLD 튜닝용 진단 스크립트.
golden_set.json의 각 질문에 대해 top-1 유사도 점수와, 정답 PMID가 top-k 안에
실제로 들어왔는지(hit)를 함께 출력한다.
API 키 없이 바로 실행 가능 (로컬 임베딩만 사용)

사용법:
    python3 check_scores.py
"""
import json
from qa_chain import get_vectorstore, SCORE_THRESHOLD, TOP_K

GOLDEN_SET_PATH = "golden_set.json"


def main():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    vectorstore = get_vectorstore()
    print(f"현재 SCORE_THRESHOLD = {SCORE_THRESHOLD}\n")

    rows = []
    for case in golden_set:
        results = vectorstore.similarity_search_with_relevance_scores(case["question"], k=TOP_K)
        score = results[0][1] if results else None
        retrieved_pmids = [d.metadata.get("pmid") for d, _ in results]
        hit = bool(set(case["expected_pmids"]) & set(retrieved_pmids))
        passes = score is not None and score >= SCORE_THRESHOLD
        rows.append((case["id"], score, hit, passes))
        mark = "OK" if passes else "BLOCKED"
        print(f"  [{case['id']}] top1_score={score:.4f}  hit={hit}  -> {mark}")

    scores = [r[1] for r in rows if r[1] is not None]
    passed = sum(1 for r in rows if r[3])
    print(f"\n{passed}/{len(rows)}개 통과 (threshold={SCORE_THRESHOLD})")
    print(f"점수 분포: min={min(scores):.4f}, max={max(scores):.4f}, "
          f"median={sorted(scores)[len(scores)//2]:.4f}")

    hit_scores = sorted(r[1] for r in rows if r[2])
    miss_scores = sorted(r[1] for r in rows if not r[2])
    print(f"\nhit=True  질문들의 점수({len(hit_scores)}개): {[round(s, 3) for s in hit_scores]}")
    print(f"hit=False 질문들의 점수({len(miss_scores)}개): {[round(s, 3) for s in miss_scores]}")


if __name__ == "__main__":
    main()
