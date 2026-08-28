"""
DPO 파인튜닝 sLLM 평가 스크립트.

Golden Set 14개(학습에 전혀 쓰이지 않은 held-out 질문)에 대해 base sLLM과
DPO 파인튜닝 sLLM의 답변을 비교합니다. 둘 다 코드 레벨 Grounding Guard 없이
"검색된 근거만으로 답하라"는 프롬프트만 준 상태로 실행해서, DPO 학습이 모델
스스로 (a) 근거 부족 시 거절하는지 (b) 답할 때 PMID를 인용하는지를 얼마나
익혔는지 측정합니다.

핵심 비교 지표:
  - self_refusal_correct: 모델이 스스로 "거절"할지 "답변"할지 판단한 게,
    기존 Grounding Guard의 threshold 기준 판단과 일치하는 비율
  - citation_rate: 답변(비거절)한 케이스 중 (PMID: xxxxx) 형식을 지킨 비율

사전 준비:
    Phase 3에서 만든 ./adapters (DPO LoRA 어댑터)가 있어야 함
    (build_index.py로 만든 ./chroma_db도 필요)

사용법:
    python3 evaluate_sllm.py
"""
import json
import re

from mlx_lm import generate, load

from qa_chain import GEN_MODEL as _  # noqa: F401 (미사용, Gemini 관련 상수와 구분하기 위해 주석용)
from qa_chain import PROMPT, SCORE_THRESHOLD, TOP_K, get_vectorstore

BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER_PATH = "adapters"
GOLDEN_SET_PATH = "golden_set.json"
OUTPUT_PATH = "sllm_evaluation_result.json"
MAX_TOKENS = 300

CITATION_PATTERN = re.compile(r"\(PMID[:\s]*\d+\)")
REFUSAL_MARKERS = [
    "확인 불가", "확인해드릴 수 없", "찾을 수 없", "포함되어 있지 않",
    "답변할 수 없", "답변을 작성할 수 없", "알 수 없습니다", "근거가 없",
]


def looks_like_refusal(text):
    return any(marker in text for marker in REFUSAL_MARKERS)


def build_prompt(context, question):
    """qa_chain.py와 동일한 지시문(PROMPT)을 재사용 -- 두 모델이 같은 조건에서
    비교되도록 프로젝트 전체에서 프롬프트를 하나로 유지."""
    messages = PROMPT.format_messages(context=context, question=question)
    return messages[0].content


def generate_answer(model, tokenizer, user_content):
    chat_messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(chat_messages, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)


def evaluate_model(model, tokenizer, cases, label):
    print(f"\n=== {label} 평가 중 ===")
    results = []
    for case in cases:
        answer = generate_answer(model, tokenizer, case["user_content"])
        is_refusal = looks_like_refusal(answer)
        has_citation = bool(CITATION_PATTERN.search(answer)) if not is_refusal else None
        self_refusal_correct = is_refusal == case["guard_would_block"]

        results.append({
            "id": case["id"],
            "answer": answer,
            "is_refusal": is_refusal,
            "has_citation": has_citation,
            "guard_would_block": case["guard_would_block"],
            "self_refusal_correct": self_refusal_correct,
        })
        mark = "OK" if self_refusal_correct else "MISMATCH"
        print(f"  [{case['id']}] guard_would_block={case['guard_would_block']} "
              f"model_refused={is_refusal} -> {mark}")

    n = len(results)
    n_self_refusal_correct = sum(r["self_refusal_correct"] for r in results)
    answered = [r for r in results if not r["is_refusal"]]
    n_cited = sum(1 for r in answered if r["has_citation"])

    summary = {
        "self_refusal_accuracy": round(n_self_refusal_correct / n, 3) if n else None,
        "n_answered": len(answered),
        "citation_rate_among_answered": round(n_cited / len(answered), 3) if answered else None,
    }
    print(f"  -> self_refusal_accuracy={summary['self_refusal_accuracy']}, "
          f"citation_rate={summary['citation_rate_among_answered']} "
          f"({n_cited}/{len(answered)})")
    return results, summary


def main():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    vectorstore = get_vectorstore()

    # 각 질문에 대해 검색 컨텍스트 + Guard가 실제로 차단했을지 여부를 미리 계산
    # (base/dpo 두 모델에게 동일한 조건을 주기 위해 한 번만 검색)
    cases = []
    for case in golden_set:
        results = vectorstore.similarity_search_with_relevance_scores(case["question"], k=TOP_K)
        top1 = results[0][1] if results else None
        guard_would_block = top1 is None or top1 < SCORE_THRESHOLD
        context = "\n\n".join(
            f"(PMID: {doc.metadata.get('pmid')}) {doc.page_content}" for doc, _ in results
        )
        cases.append({
            "id": case["id"],
            "user_content": build_prompt(context, case["question"]),
            "top1_score": top1,
            "guard_would_block": guard_would_block,
        })

    print("=== Base 모델 로딩 ===")
    base_model, base_tokenizer = load(BASE_MODEL)
    base_results, base_summary = evaluate_model(base_model, base_tokenizer, cases, "Base (파인튜닝 전)")

    print("\n=== DPO 파인튜닝 모델 로딩 ===")
    dpo_model, dpo_tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
    dpo_results, dpo_summary = evaluate_model(dpo_model, dpo_tokenizer, cases, "DPO 파인튜닝 후")

    output = {
        "base_summary": base_summary,
        "dpo_summary": dpo_summary,
        "base_results": base_results,
        "dpo_results": dpo_results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== 비교 요약 ===")
    print(f"{'지표':<28}{'Base':>10}{'DPO':>10}")
    print(f"{'self_refusal_accuracy':<28}{base_summary['self_refusal_accuracy']!s:>10}"
          f"{dpo_summary['self_refusal_accuracy']!s:>10}")
    print(f"{'citation_rate':<28}{base_summary['citation_rate_among_answered']!s:>10}"
          f"{dpo_summary['citation_rate_among_answered']!s:>10}")
    print(f"\n-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
