"""
preference_data.jsonl의 prompt를 재구성합니다.

문제: 기존 prompt 필드는 질문 텍스트뿐이라, 실제 추론/평가 시 쓰는 프롬프트 형식
(qa_chain.PROMPT -- "지시문 + 검색된 context + 질문")과 완전히 달랐습니다. 즉 DPO 학습
시점의 입력 형식과 실사용 시점의 입력 형식이 달라서, 학습된 선호가 실사용 프롬프트에
전이되지 않았을 가능성이 있습니다 (evaluate_sllm.py는 항상 context를 포함한 프롬프트로
평가함).

이 스크립트는 API 호출 없이(로컬 벡터스토어 검색만 재실행) 기존 질문에 대해 context를
복원하고, qa_chain.PROMPT와 동일한 형식으로 prompt를 재구성합니다. 동시에 weak_evidence
시나리오의 chosen 답변도 고정된 문구 1개 대신 여러 자연스러운 거절 표현으로 다양화합니다
(evaluate_sllm.py의 REFUSAL_MARKERS와 호환되도록 각 표현에 마커 문자열을 포함시킴).

사용법:
    python3 rebuild_preference_prompts.py
"""
import json

from qa_chain import PROMPT, TOP_K, format_context, get_vectorstore

INPUT_PATH = "preference_data_bare_question.jsonl"  # 백업된 원본 (prompt=질문 텍스트만)
OUTPUT_PATH = "preference_data.jsonl"  # context 포함 형식으로 재구성한 결과

# evaluate_sllm.py의 REFUSAL_MARKERS와 호환되도록, 각 문장이 마커 문자열을 하나 이상 포함
REFUSAL_VARIANTS = [
    "관련 근거 논문을 찾을 수 없어 확인해드릴 수 없습니다.",
    "제공된 자료만으로는 이 질문에 답변할 수 없습니다.",
    "이 질문에 대한 근거가 없어 답변을 작성할 수 없습니다.",
    "관련 내용이 포함되어 있지 않아 확인이 어렵습니다.",
    "현재로서는 확인 불가한 내용입니다.",
    "해당 질문에 답할 근거 논문을 찾을 수 없습니다.",
    "정확한 근거가 없어 답변드리기 어렵습니다.",
    "관련 자료가 부족하여 답변할 수 없습니다.",
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_full_prompt(question, vectorstore):
    results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
    context = format_context(results) if results else "(검색된 자료 없음)"
    messages = PROMPT.format_messages(context=context, question=question)
    return messages[0].content


def main():
    pairs = load_jsonl(INPUT_PATH)
    vectorstore = get_vectorstore()

    weak_i = 0
    for p in pairs:
        original_question = p["prompt"]  # 지금까지는 이게 곧 질문 텍스트였음
        p["question"] = original_question  # 원본 질문도 보존 (참고/디버깅용)
        p["prompt"] = build_full_prompt(original_question, vectorstore)

        if p["scenario"] == "weak_evidence":
            p["chosen"] = REFUSAL_VARIANTS[weak_i % len(REFUSAL_VARIANTS)]
            weak_i += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_grounded = sum(1 for p in pairs if p["scenario"] == "grounded")
    n_weak = sum(1 for p in pairs if p["scenario"] == "weak_evidence")
    print(f"완료: {len(pairs)}개 재구성 (grounded {n_grounded} + weak_evidence {n_weak})")
    print(f"-> {OUTPUT_PATH} (prompt에 context 포함, chosen 거절 문구 {len(REFUSAL_VARIANTS)}종 순환 배정)")


if __name__ == "__main__":
    main()
