"""
SCORE_THRESHOLD 재튜닝용 보조 진단 스크립트 (임베딩 모델 교체 후, 2026-10-08).

문제: check_scores.py의 golden set(14개) 결과를 보면, hit=False인 케이스(q-012)가
hit=True인 여러 케이스보다 오히려 top1_score가 더 높게 나옴 — "점수가 높다"는 게
"정답 PMID를 찾았다"는 뜻이 아니라 "주제상 관련 있는 문서를 찾았다"는 뜻일 뿐이라는
의미. 게다가 golden set 14개 전부 점수가 0.49 이상이라, threshold=0.30이 지금은
아무것도 못 거르는 상태(14/14 통과) — Guard를 제대로 튜닝하려면 "진짜 근거가 없어서
막아야 하는" 질문의 점수가 필요한데 golden set에는 그런 예시가 없음.

이 스크립트는 코퍼스(암 유전체/전사체 PubMed 초록) 범위 밖의 질문들을 넣어서 top1_score가
실제로 낮게 나오는지 확인한다. 이 "밖" 범위의 점수와 golden set "안" 범위의 점수(최소 0.49)
사이 어딘가로 threshold를 잡으면 된다. 우선순위 4(Secure RAG)에서 만들 적대적 테스트셋의
초기 버전이기도 함.

사용법:
    python check_offtopic_scores.py
"""
from qa_chain import get_vectorstore, TOP_K

# 코퍼스 범위 밖 질문들. 완전히 무관한 것(1~2)과, 의학이지만 이 코퍼스(암 유전체) 밖
# 주제인 것(3~4, "그럴듯하게 들리지만 실제로는 근거가 없는" adversarial 케이스)을 섞음.
OFFTOPIC_QUESTIONS = [
    "서울시 지하철 요금 체계는 어떻게 구성되어 있나요?",
    "오늘 저녁 메뉴로 뭘 추천해?",
    "2026년 월드컵 개최국은 어디인가?",
    "제2형 당뇨병의 인슐린 저항성 기전은 무엇인가?",
]


def main():
    vectorstore = get_vectorstore()

    print("=== 코퍼스 밖(off-topic) 질문들의 top1_score ===\n")
    offtopic_scores = []
    for q in OFFTOPIC_QUESTIONS:
        results = vectorstore.similarity_search_with_relevance_scores(q, k=TOP_K)
        score = results[0][1] if results else None
        top1_title = results[0][0].metadata.get("title", "")[:60] if results else "-"
        offtopic_scores.append(score)
        print(f"  Q: {q}")
        print(f"     top1_score={score:.4f}  (가장 가까웠던 문서: {top1_title}...)\n")

    print(f"off-topic 점수 범위: min={min(offtopic_scores):.4f}, max={max(offtopic_scores):.4f}")
    print("(참고: golden set 14개 in-domain 질문의 최소 점수는 0.4919였음 — check_scores.py 결과)")


if __name__ == "__main__":
    main()
