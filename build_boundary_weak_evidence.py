"""
weak_evidence "경계선 예시" 재구성 스크립트.

기존 weak_evidence(89개 이종 도메인 주제 -- 코퍼스 4개 도메인과 무관한 암종이라 표면적
어휘만으로도 grounded와 구분되는 "너무 쉬운" 문제였음)를, 코퍼스와 **같은 4개 도메인이지만
색인되지 않은** 논문(data/abstracts_holdout.json, fetch_pubmed_holdout.py로 수집)에서
만든 질문으로 교체합니다. 같은 도메인 어휘를 쓰지만 실제 검색 시 근거가 안 잡히는 "진짜
경계선 사례"라 DPO가 표면적 주제 분류로 지름길을 타지 못하게 하는 게 목적입니다.

grounded 112개(이미 context 포함 prompt로 재구성된 상태)는 그대로 유지하고 weak_evidence만
교체합니다.

사전 준비:
    export GOOGLE_API_KEY="..."
    python3 fetch_pubmed_holdout.py 먼저 실행 (data/abstracts_holdout.json 필요)

사용법:
    python3 build_boundary_weak_evidence.py
"""
import json
import os
import random
import time

from langchain_google_genai import ChatGoogleGenerativeAI

from build_preference_data import (
    GEN_MODEL,
    build_question_llm,
    generate_question_from_abstract,
    make_ungrounded_answer,
)
from qa_chain import SCORE_THRESHOLD, TOP_K, get_vectorstore
from rebuild_preference_prompts import REFUSAL_VARIANTS, build_full_prompt

HOLDOUT_PATH = "data/abstracts_holdout.json"
CURRENT_PATH = "preference_data.jsonl"  # 기존 grounded 112개를 유지하기 위해 읽음
OUTPUT_PATH = "preference_data.jsonl"
SLEEP_BETWEEN_CALLS_SEC = 3

random.seed(42)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("GOOGLE_API_KEY가 설정되어 있지 않습니다. export GOOGLE_API_KEY=... 후 다시 실행하세요.")
        return

    with open(HOLDOUT_PATH, "r", encoding="utf-8") as f:
        holdout_abstracts = json.load(f)

    existing = load_jsonl(CURRENT_PATH)
    grounded_pairs = [p for p in existing if p["scenario"] == "grounded"]
    print(f"기존 grounded {len(grounded_pairs)}개는 그대로 유지")

    vectorstore = get_vectorstore()
    question_llm = build_question_llm()
    gen_llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)

    weak_pairs = []
    weak_i = 0
    print(f"\n[경계선 weak_evidence] {len(holdout_abstracts)}개 미색인 논문에서 질문 생성 중...")
    for i, abstract in enumerate(holdout_abstracts):
        try:
            question = generate_question_from_abstract(abstract, question_llm)
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
            top1 = results[0][1] if results else None
            if top1 is not None and top1 >= SCORE_THRESHOLD:
                print(f"  [{i}] 예상과 달리 검색 점수가 높아 스킵(score={top1:.3f}): {question[:40]}...")
                continue

            full_prompt = build_full_prompt(question, vectorstore)
            chosen = REFUSAL_VARIANTS[weak_i % len(REFUSAL_VARIANTS)]
            weak_i += 1
            rejected = make_ungrounded_answer(question, gen_llm)
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            weak_pairs.append({
                "prompt": full_prompt,
                "question": question,
                "chosen": chosen,
                "rejected": rejected,
                "scenario": "weak_evidence",
                "source_pmid": abstract["pmid"],  # 참고용 -- 미색인 논문이라 실제 근거로는 안 씀
            })
            score_str = f"{top1:.3f}" if top1 is not None else "None"
            print(f"  [{i}] OK (score={score_str}, 코퍼스 도메인={abstract.get('topic','')}): {question[:40]}...")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] 실패, 스킵: {e}")

    all_pairs = grounded_pairs + weak_pairs
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n완료: grounded {len(grounded_pairs)}개 + weak_evidence(경계선) {len(weak_pairs)}개 = 총 {len(all_pairs)}개")
    print(f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
