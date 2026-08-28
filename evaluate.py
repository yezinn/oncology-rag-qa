"""
Golden Set 기반 평가 스크립트

1) Retrieval Hit Rate: 기대 PMID가 실제로 검색되었는지 자동 측정 (기존 로직, 변경 없음)
2) LLM-as-a-Judge: qa_chain.py로 실제 답변을 생성시킨 뒤, 별도의 LLM 채점자가
   (a) Faithfulness — 답변의 모든 주장이 제공된 근거 초록에서 실제로 확인되는가
   (b) Relevance — 답변이 질문을 실제로 다루고 있는가
   두 기준으로 1~5점을 매김. GOOGLE_API_KEY가 설정된 경우에만 실행되고,
   없으면 1)만 실행하고 건너뜀 (키 없이도 기존처럼 정상 동작).

사전 준비:
    golden_set_template.json을 복사해 golden_set.json으로 만들고, 본인 지식으로
    질문과 정답 PMID를 채워넣는다(최소 10~15개 권장).
    LLM-as-a-Judge까지 실행하려면: export GOOGLE_API_KEY="..."
    (Google AI Studio 발급: https://aistudio.google.com/apikey)

사용법:
    python evaluate.py
"""
import json
import os
import time
from qa_chain import get_vectorstore, answer_question, TOP_K

GOLDEN_SET_PATH = "golden_set.json"
# gemini-2.5-flash-lite는 신규 사용자에게 더 이상 제공되지 않아(2026-08 기준)
# gemini-3.5-flash-lite로 교체함. 무료 티어 분당 요청 제한에 여유를 두기 위해 호출 간 슬립도 유지.
JUDGE_MODEL = "gemini-3.5-flash-lite"
SLEEP_BETWEEN_CALLS_SEC = 2

JUDGE_PROMPT = """당신은 RAG(검색 기반 질의응답) 시스템의 답변 품질을 평가하는 엄격한 채점자입니다.
아래 [질문]에 대해 시스템이 생성한 [답변]을, 시스템이 근거로 사용한 [제공된 근거 초록]만을
기준으로 평가하세요. 초록에 없는 내용을 사실로 단정했다면 faithfulness를 낮게 주세요.

[질문]
{question}

[제공된 근거 초록]
{context}

[생성된 답변]
{answer}

다음 두 기준으로 1~5점(정수)으로 채점하세요:
- faithfulness: 답변의 모든 주장이 [제공된 근거 초록]에서 실제로 확인되는가?
- relevance: 답변이 [질문]에서 실제로 묻는 내용을 다루고 있는가?
"""


def build_judge_llm():
    """구조화 출력(JSON) 강제를 위해 Pydantic 스키마로 with_structured_output 사용."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import BaseModel, Field

    class JudgeScore(BaseModel):
        faithfulness: int = Field(description="1~5점: 답변의 모든 주장이 근거 초록에서 실제로 확인되는가")
        relevance: int = Field(description="1~5점: 답변이 질문에서 실제로 묻는 내용을 다루는가")
        reasoning: str = Field(description="채점 근거를 한 문장으로")

    base_llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0)
    return base_llm.with_structured_output(JudgeScore)


def judge_answer(question, context, answer, structured_judge_llm):
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    try:
        result = structured_judge_llm.invoke(prompt)
        return {
            "faithfulness": result.faithfulness,
            "relevance": result.relevance,
            "reasoning": result.reasoning,
        }
    except Exception as e:  # noqa: BLE001 - 채점 실패는 스킵하고 계속 진행
        return {"faithfulness": None, "relevance": None, "reasoning": f"채점 실패: {e}"}


def main():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    vectorstore = get_vectorstore()

    run_judge = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    gen_llm = judge_llm = None
    if run_judge:
        from qa_chain import GEN_MODEL
        from langchain_google_genai import ChatGoogleGenerativeAI

        gen_llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)
        judge_llm = build_judge_llm()
    else:
        print("(GOOGLE_API_KEY 미설정 — LLM-as-a-Judge는 건너뛰고 retrieval hit rate만 측정합니다)\n")

    hits = 0
    results_log = []
    faithfulness_scores, relevance_scores = [], []

    for case in golden_set:
        docs = vectorstore.similarity_search(case["question"], k=TOP_K)
        retrieved_pmids = [d.metadata.get("pmid") for d in docs]
        expected = set(case["expected_pmids"])
        hit = bool(expected & set(retrieved_pmids))
        hits += int(hit)

        entry = {
            "id": case["id"],
            "question": case["question"],
            "expected_pmids": case["expected_pmids"],
            "retrieved_pmids": retrieved_pmids,
            "hit": hit,
        }

        if run_judge:
            result = answer_question(case["question"], gen_llm, vectorstore)
            entry["generated_answer"] = result["answer"]
            entry["grounded"] = result["grounded"]
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            if result["grounded"]:
                context = "\n\n".join(
                    f"(PMID: {d.metadata.get('pmid')}) {d.page_content}" for d in docs
                )
                judge = judge_answer(case["question"], context, result["answer"], judge_llm)
                time.sleep(SLEEP_BETWEEN_CALLS_SEC)
            else:
                judge = {
                    "faithfulness": None,
                    "relevance": None,
                    "reasoning": "Grounding Guard가 근거 부족으로 답변 생성을 차단함",
                }
            entry["llm_judge"] = judge

            if judge["faithfulness"] is not None:
                faithfulness_scores.append(judge["faithfulness"])
                relevance_scores.append(judge["relevance"])

            print(f"  [{case['id']}] hit={hit} | faithfulness={judge['faithfulness']} relevance={judge['relevance']}")
        else:
            print(f"  [{case['id']}] hit={hit}")

        results_log.append(entry)

    hit_rate = hits / len(golden_set) if golden_set else 0
    print(f"\nRetrieval Hit Rate: {hit_rate:.2f} ({hits}/{len(golden_set)})")

    output = {"hit_rate": hit_rate, "cases": results_log}

    if run_judge and faithfulness_scores:
        output["llm_judge_summary"] = {
            "avg_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 2),
            "avg_relevance": round(sum(relevance_scores) / len(relevance_scores), 2),
            "n_judged": len(faithfulness_scores),
            "n_total": len(golden_set),
        }
        print(f"평균 Faithfulness: {output['llm_judge_summary']['avg_faithfulness']} / 5")
        print(f"평균 Relevance: {output['llm_judge_summary']['avg_relevance']} / 5")

    with open("evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n상세 결과 -> evaluation_result.json")


if __name__ == "__main__":
    main()
