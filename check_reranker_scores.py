"""
Reranker(Cross-Encoder) 도입 가설 검증용 진단 스크립트 (2026-10-08).

배경: §5(skala-rag-lecture-notes.md)에서 dense cosine 임계값 하나로는
"완전히 무관한 주제"는 잘 걸러도(score 0.16~0.19), "의학적으로 그럴듯하지만
코퍼스 밖 도메인"인 질문(제2형 당뇨병 인슐린 저항성 — 이 코퍼스는 암 유전체
전문)은 오히려 진짜 정답(최소 0.49)보다도 높은 score(0.71)가 나와서 threshold로
못 거른다는 구조적 한계가 확인됨.

가설: Cross-Encoder는 질문과 문서를 함께(concatenate해서) 평가하므로, 단순
벡터 거리보다 세부 도메인 불일치를 더 잘 잡아낼 수 있을 것이다.

다국어 reranker 모델(BAAI/bge-reranker-v2-m3) 사용 — 영어 전용 모델을 썼다면
한국어 질문에서 §5와 같은 cross-lingual 문제를 reranker 단계에서 반복했을 것.

사용법:
    python check_reranker_scores.py
    (처음 실행 시 reranker 모델 다운로드 — all-MiniLM보다 큼, 몇 분 걸릴 수 있음)
"""
import json
from qa_chain import get_vectorstore, RERANKER_MODEL, RERANK_CANDIDATE_K
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

GOLDEN_SET_PATH = "golden_set.json"

# check_offtopic_scores.py와 동일한 질문 — dense cosine 결과와 직접 비교하기 위함
OFFTOPIC_QUESTIONS = [
    "서울시 지하철 요금 체계는 어떻게 구성되어 있나요?",
    "오늘 저녁 메뉴로 뭘 추천해?",
    "2026년 월드컵 개최국은 어디인가?",
]
# dense cosine에서 0.7098로 '통과 위험' 확인된 adversarial 케이스
ADVERSARIAL_QUESTIONS = [
    "제2형 당뇨병의 인슐린 저항성 기전은 무엇인가?",
]


def top1_rerank(cross_encoder, vectorstore, question, expected_pmid=None):
    candidates = vectorstore.similarity_search(question, k=RERANK_CANDIDATE_K)
    if not candidates:
        return None, None, False
    pairs = [(question, d.page_content) for d in candidates]
    scores = list(cross_encoder.score(pairs))
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    best_doc, best_score = candidates[best_i], scores[best_i]
    hit = expected_pmid is not None and best_doc.metadata.get("pmid") == expected_pmid
    return best_score, best_doc.metadata.get("pmid"), hit


def main():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    vectorstore = get_vectorstore()
    print(f"Reranker 모델 로딩 중: {RERANKER_MODEL} (처음이면 다운로드)")
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)

    print("\n=== Golden set (in-domain, 14개) ===")
    in_domain_scores = []
    for case in golden_set:
        score, pmid, hit = top1_rerank(
            cross_encoder, vectorstore, case["question"], case["expected_pmids"][0]
        )
        in_domain_scores.append(score)
        print(f"  [{case['id']}] rerank_top1_score={score:.4f}  hit={hit}")

    print("\n=== 완전히 무관한 질문 ===")
    offtopic_scores = []
    for q in OFFTOPIC_QUESTIONS:
        score, _, _ = top1_rerank(cross_encoder, vectorstore, q)
        offtopic_scores.append(score)
        print(f"  Q: {q}\n     rerank_top1_score={score:.4f}")

    print("\n=== 의학적으로 그럴듯하지만 다른 도메인 (adversarial) ===")
    adversarial_scores = []
    for q in ADVERSARIAL_QUESTIONS:
        score, _, _ = top1_rerank(cross_encoder, vectorstore, q)
        adversarial_scores.append(score)
        print(f"  Q: {q}\n     rerank_top1_score={score:.4f}")

    print(f"\nin-domain 최소: {min(in_domain_scores):.4f}")
    print(f"off-topic 최대: {max(offtopic_scores):.4f}")
    print(f"adversarial 최대: {max(adversarial_scores):.4f}")
    print("\n(참고: dense cosine에서는 in-domain 최소 0.4919 / off-topic 최대 0.1851 / adversarial 0.7098)")
    print("-> adversarial이 in-domain 최소보다 낮아지면 reranker가 dense cosine보다 도메인 구분을 잘 하는 것")


if __name__ == "__main__":
    main()
