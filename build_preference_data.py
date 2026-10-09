"""
DPO 학습용 Preference 데이터셋 구축 스크립트.

기존 RAG 파이프라인(qa_chain.py)과 인덱스(chroma_db)를 재사용해서, 사람이 일일이
preference pair를 작성하는 대신 두 가지 시나리오로 프로그램적으로 대조쌍(chosen/rejected)을
생성합니다. (사람이 1:1로 라벨링한 preference 데이터는 아니며, README에도 그렇게 명시할 것)

1) 근거 충분 시나리오: 실제 색인된 초록에서 질문을 만들고,
   - chosen   = RAG 파이프라인이 실제로 생성한 근거 기반 답변 (PMID 인용 포함)
   - rejected = 같은 질문을 검색 없이 LLM에 직접 물어본 답변 (근거 없는 맨몸 답변)

2) 근거 불충분 시나리오 (Grounding Guard가 차단하는 상황을 재현):
   - 색인된 4개 토픽(EGFR LUAD / TNBC 항암화학 / ssGSEA / 전이학습 약물반응)과
     무관한 암 유전체 인접 주제로 질문을 만들어 검색 점수가 낮게 나오도록 유도
   - chosen   = "확인 불가" 정직한 거절 답변 (qa_chain.py의 Guard 문구와 동일)
   - rejected = 검증 절차 없이 곧바로 질문에 답하게 한 결과 (make_ungrounded_answer 재사용)
     -> 최초 시도: 약한 근거를 주고 Guard를 무시하라고 프롬프트로 강제(make_forced_answer_
        ignoring_guard)했으나, Gemini가 프롬프트 지시에도 불구하고 "초록에 근거가 없다"며
        스스로 재차 거절하는 경우가 대부분이었음(12개 전량). chosen과 rejected가 의미상
        거의 동일해져 preference 신호로 쓸 수 없다고 판단, 컨텍스트 자체를 주지 않고 바로
        답하게 하는 방식(시나리오 1의 rejected와 동일 함수)으로 전환함.

주의: 여기서 쓰는 질문 풀은 golden_set.json(평가용 14개)이 정답으로 쓰는 PMID를
      제외하고 샘플링합니다 -- 학습 데이터와 평가 데이터가 섞이면 이후 sLLM 평가
      결과를 신뢰할 수 없게 됩니다.

사전 준비:
    export GOOGLE_API_KEY="..."
    (build_index.py로 만든 ./chroma_db가 있어야 함)

사용법:
    python3 build_preference_data.py                        # 두 시나리오 모두 실행
    python3 build_preference_data.py --scenario grounded
    python3 build_preference_data.py --scenario weak_evidence

주의(무료 티어 quota): Gemini 무료 티어는 하루 요청 수가 제한(모델당 500회)됩니다.
grounded 시나리오만 해도 시도 300개 * (질문 1회 + 통과 시 답변 2회)로 500회에
근접/초과할 수 있습니다. quota 초과 에러가 뜨면 나머지 호출도 전부 실패할 게
뻔하므로 스크립트가 즉시 중단하고 그때까지 모은 pair를 저장합니다. --scenario로
시나리오를 나눠서 날짜를 나눠 실행하세요. 기존 preference_data.jsonl은 지우지 않고
"이번에 실행한 시나리오"만 교체해서 병합합니다.
"""
import argparse
import json
import os
import random
import time

from qa_chain import (
    GEN_MODEL,
    PROMPT,
    SCORE_THRESHOLD,
    TOP_K,
    answer_question,
    extract_text,
    format_context,
    get_vectorstore,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

ABSTRACTS_PATH = "data/abstracts.json"
GOLDEN_SET_PATH = "golden_set.json"
OUTPUT_PATH = "preference_data.jsonl"

N_GROUNDED_TARGET = 300   # 근거 충분 시나리오에서 시도할 초록 개수 (일부는 필터링으로 탈락 가능. 코퍼스 확장(240->~500건) 후 목표를 40->300으로 상향)
SLEEP_BETWEEN_CALLS_SEC = 3  # 무료 티어 분당 요청 제한 대비. 레이트리밋 에러 나면 늘리세요.

random.seed(42)  # 어떤 초록이 샘플링됐는지 재현 가능해야 하므로 고정

# 색인된 4개 토픽과 무관한, 암 유전체/전사체 인접 주제 -- "근거 불충분" 상황을 의도적으로 유도.
WEAK_EVIDENCE_TOPICS = [
    "췌장암 면역항암요법 저항성 기전",
    "대장암 CRISPR 기능유전체 스크리닝",
    "교모세포종 단일세포 공간전사체 분석",
    "전립선암 안드로겐 수용체 스플라이스 변이체",
    "난소암 BRCA 변이 PARP 저해제 반응",
    "다발골수종 미세잔존질환 액체생검 검출",
    "간세포암 면역관문억제제 병용요법",
    "급성골수성백혈병 미세잔존질환 분자 모니터링",
    "신세포암 면역관문억제제 반응 바이오마커",
    "흑색종 BRAF 표적치료 내성 기전",
    "위암 HER2 표적치료 반응 예측",
    "식도암 방사선-면역요법 병용 반응",
    "두경부암 PD-L1 발현과 면역치료 반응",
    "방광암 FGFR 변이 표적치료",
    "육종 면역세포 침윤과 예후",
    "자궁경부암 HPV 통합 부위 분석",
    "자궁내막암 분자아형 분류",
    "갑상선암 BRAF V600E 표적치료 반응",
    "소포성 림프종 후성유전학적 조절",
    "만성림프구성백혈병 미세잔존질환 액체생검",
    "급성림프모구백혈병 융합유전자 검출",
    "중피종 면역관문억제제 반응 바이오마커",
    "신경내분비종양 소마토스타틴 수용체 발현",
    "담관암 FGFR2 융합유전자 표적치료",
    "고환암 화학요법 저항성 기전",
    "신우요관암 분자프로파일링",
    "음경암 HPV 관련 발암기전",
    "외음부암 면역세포 침윤 분석",
    "구강편평세포암 종양미세환경 특성",
    "비인두암 EBV 바이러스 부하와 예후",
    "후두암 방사선치료 반응 예측인자",
    "흉선종 자가면역 연관 유전자 발현",
    "부신피질암 스테로이드 생합성 경로",
    "위장관기질종양 KIT 변이 표적치료 내성",
    "망막모세포종 RB1 유전자 불활성화 기전",
    "윌름스종양 소아 예후 바이오마커",
    "골육종 폐전이 관련 유전자 발현",
    "유잉육종 EWSR1 융합유전자 검출",
    "횡문근육종 근원성 분화 마커",
    "간모세포종 베타카테닌 경로 활성화",
    "신경모세포종 MYCN 증폭과 예후",
    "수모세포종 분자아형별 치료 반응",
    "척삭종 브라키유리 유전자 발현",
    "균상식육종 T세포 클론성 분석",
    "메르켈세포암 폴리오마바이러스 연관성",
    "선양낭성암 MYB 유전자 재배열",
    "담낭암 면역관문억제제 병용요법 반응",
    "부비동암 종양 침습성 유전자 프로파일",
    "눈꺼풀샘암 조직학적 아형 분류",
    "침샘암 안드로겐 수용체 발현과 치료반응",
    "자궁육종 재발 관련 바이오마커",
    "난관암 BRCA 변이 스크리닝",
    "외분비췌장암 오가노이드 약물 스크리닝",
    "담도암 다중오믹스 통합 분석",
    "부신수질암 카테콜아민 대사 경로",
    "흉막암 엑소좀 기반 조기진단 바이오마커",
    "복막중피종 유전체 불안정성 특성",
    "원발불명암 전이 조직 기원 예측",
    "갑상선수질암 RET 유전자 변이 표적치료",
    "부갑상선암 칼슘 대사 관련 유전자",
    "흉선암 면역치료 반응 예측",
    "성인T세포백혈병 HTLV-1 바이러스 연관 발암기전",
    "모발세포백혈병 BRAF 변이 표적치료",
    "혈관육종 방사선 유발 발암기전",
    "지방육종 MDM2 유전자 증폭",
    "활막육종 SS18-SSX 융합유전자",
    "평활근육종 재발 예측 바이오마커",
    "섬유육종 종양억제유전자 불활성화",
    "신장혈관근지방종 mTOR 경로 활성화",
    "방광상피내암 BCG 치료 반응 예측",
    "요관암 분자아형별 화학요법 반응",
    "음낭육종 유전자 발현 프로파일",
    "질암 HPV 관련 분자병리",
    "자궁경부신경내분비암종 치료 저항성",
    "난소경계성종양 재발 예측 유전자",
    "자궁내막간질육종 유전자 재배열",
    "위선암 미세부수체 불안정성과 면역치료",
    "소장선암 유전체 프로파일링",
    "항문암 HPV 백신과 발암 예방",
    "직장신경내분비종양 소마토스타틴 유사체 반응",
    "담낭선암 조기진단 액체생검 바이오마커",
    "간내담관암 IDH1 변이 표적치료",
    "폐신경내분비암 소세포/비소세포 감별 마커",
    "흉선상피종양 WHO 분류별 예후인자",
    "비소세포폐암 KRAS G12C 표적치료 반응",
    "소세포폐암 DLL3 표적 항체약물접합체",
    "중추신경계 림프종 혈액뇌장벽 투과 약물반응",
    "뇌하수체선종 호르몬 분비 관련 유전자",
    "척수종양 방사선치료 후 재발 바이오마커",
]


class GeneratedQuestion(BaseModel):
    question: str = Field(description="구체적이고 사실 확인 가능한 연구 질문 (한국어)")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_question_llm():
    llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0.7)
    return llm.with_structured_output(GeneratedQuestion)


def generate_question_from_abstract(abstract, question_llm):
    prompt = f"""다음은 PubMed 논문 초록입니다. 이 초록의 내용만으로 명확히 답할 수 있는,
구체적이고 사실 확인 가능한 연구 질문을 한국어로 1개 만드세요. 너무 일반적인 질문(예: "이
논문의 주제는?")은 피하고, 초록 속 구체적인 수치/유전자/기전을 묻는 질문을 만드세요.

[제목] {abstract['title']}
[초록] {abstract['abstract']}
"""
    result = question_llm.invoke(prompt)
    return result.question


def generate_weak_evidence_question(topic, question_llm):
    prompt = f"""당신은 암 유전체/전사체 연구자입니다. 최근 "{topic}" 분야에서 나올 법한,
구체적이고 자연스러운 연구 질문을 한국어로 1개 만드세요. 실제 논문에 답이 있을 법한
질문이면 됩니다 (당신이 그 답을 알 필요는 없습니다)."""
    result = question_llm.invoke(prompt)
    return result.question


def make_ungrounded_answer(question, gen_llm):
    """검색 없이 LLM에 직접 물어본, 근거/인용 없는 답변 (rejected 후보)."""
    prompt = f"""다음 질문에 대해 알고 있는 지식을 바탕으로 답하세요. 출처나 PMID를
인용하지 마세요.

[질문] {question}"""
    response = gen_llm.invoke(prompt)
    return extract_text(response)


def make_forced_answer_ignoring_guard(question, results, gen_llm):
    """근거가 약해도 Grounding Guard를 무시하고 억지로 생성시킨 답변 (rejected 후보).

    주의: 실제로는 사용하지 않음. Gemini가 프롬프트로 강제해도 스스로 거절하는 경우가
    대부분이라(12/12), chosen과 구분이 안 되는 무의미한 preference 쌍이 됨. 대신
    make_ungrounded_answer()로 대체함 (아래 main() 참고). 무엇을 시도했고 왜 안 됐는지
    남겨두기 위해 함수는 그대로 둠."""
    context = format_context(results)
    chain = PROMPT | gen_llm
    response = chain.invoke({"context": context, "question": question})
    return extract_text(response)


class QuotaExceeded(Exception):
    """Gemini API 일일/분당 quota 초과 (429). 재시도해도 소용없으므로 즉시 중단시키기 위한 예외."""


def is_quota_error(e):
    msg = str(e)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg or "quota" in msg.lower()


def load_existing_pairs(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_pairs(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def run_grounded_scenario(vectorstore, question_llm, gen_llm):
    abstracts = load_json(ABSTRACTS_PATH)
    golden_set = load_json(GOLDEN_SET_PATH)
    golden_pmids = set()
    for case in golden_set:
        golden_pmids.update(case["expected_pmids"])

    # golden_set.json이 정답으로 쓰는 논문은 학습 데이터에서 제외 (train/eval 오염 방지)
    eligible_abstracts = [a for a in abstracts if a["pmid"] not in golden_pmids]
    random.shuffle(eligible_abstracts)
    sampled_abstracts = eligible_abstracts[:N_GROUNDED_TARGET]

    pairs = []
    print(f"[근거 충분 시나리오] {len(sampled_abstracts)}개 초록에서 질문 생성 중...")
    for i, abstract in enumerate(sampled_abstracts):
        try:
            question = generate_question_from_abstract(abstract, question_llm)
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
            if not results or results[0][1] < SCORE_THRESHOLD:
                score = results[0][1] if results else None
                print(f"  [{i}] 검색 점수가 낮아 스킵(score={score}): {question[:40]}...")
                continue

            chosen = answer_question(question, gen_llm, vectorstore)["answer"]
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)
            rejected = make_ungrounded_answer(question, gen_llm)
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            pairs.append({
                "prompt": question,
                "chosen": chosen,
                "rejected": rejected,
                "scenario": "grounded",
                "source_pmid": abstract["pmid"],
            })
            print(f"  [{i}] OK: {question[:40]}...")
        except Exception as e:  # noqa: BLE001
            if is_quota_error(e):
                print(f"  [{i}] quota 초과로 중단: {e}")
                raise QuotaExceeded(str(e)) from e
            print(f"  [{i}] 실패, 스킵: {e}")
    return pairs


def run_weak_evidence_scenario(vectorstore, question_llm, gen_llm):
    pairs = []
    print(f"[근거 불충분 시나리오] {len(WEAK_EVIDENCE_TOPICS)}개 주제에서 질문 생성 중...")
    for i, topic in enumerate(WEAK_EVIDENCE_TOPICS):
        try:
            question = generate_weak_evidence_question(topic, question_llm)
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
            top1 = results[0][1] if results else None
            if top1 is not None and top1 >= SCORE_THRESHOLD:
                print(f"  [{i}] 예상과 달리 검색 점수가 높아 스킵(score={top1:.3f}): {question[:40]}...")
                continue

            chosen = "관련 근거 논문을 찾을 수 없어 확인해드릴 수 없습니다."
            rejected = make_ungrounded_answer(question, gen_llm)
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            pairs.append({
                "prompt": question,
                "chosen": chosen,
                "rejected": rejected,
                "scenario": "weak_evidence",
                "source_pmid": None,
            })
            score_str = f"{top1:.3f}" if top1 is not None else "None"
            print(f"  [{i}] OK (score={score_str}): {question[:40]}...")
        except Exception as e:  # noqa: BLE001
            if is_quota_error(e):
                print(f"  [{i}] quota 초과로 중단: {e}")
                raise QuotaExceeded(str(e)) from e
            print(f"  [{i}] 실패, 스킵: {e}")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=["grounded", "weak_evidence", "both"], default="both",
        help="실행할 시나리오. 무료 티어 quota 제한 때문에 나눠서 실행할 때 사용 (기본: both)",
    )
    args = parser.parse_args()

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("GOOGLE_API_KEY가 설정되어 있지 않습니다. export GOOGLE_API_KEY=... 후 다시 실행하세요.")
        return

    vectorstore = get_vectorstore()
    question_llm = build_question_llm()
    gen_llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)

    # 기존 결과 로드 -- 이번에 실행하지 않는 시나리오의 pair는 그대로 보존
    existing = load_existing_pairs(OUTPUT_PATH)
    run_grounded = args.scenario in ("grounded", "both")
    run_weak = args.scenario in ("weak_evidence", "both")

    kept = [p for p in existing if not (
        (run_grounded and p["scenario"] == "grounded")
        or (run_weak and p["scenario"] == "weak_evidence")
    )]
    new_pairs = []

    try:
        if run_grounded:
            new_pairs += run_grounded_scenario(vectorstore, question_llm, gen_llm)
        if run_weak:
            new_pairs += run_weak_evidence_scenario(vectorstore, question_llm, gen_llm)
    except QuotaExceeded:
        print("\n*** Gemini API quota 초과로 중단했습니다. 그때까지 모은 결과는 저장합니다. ***")
        print("*** quota가 리셋된 후 동일한 --scenario로 다시 실행하면 이어서 채울 수 있습니다. ***")
    finally:
        all_pairs = kept + new_pairs
        save_pairs(OUTPUT_PATH, all_pairs)

        n_grounded = sum(1 for p in all_pairs if p["scenario"] == "grounded")
        n_weak = sum(1 for p in all_pairs if p["scenario"] == "weak_evidence")
        print(f"\n현재 {OUTPUT_PATH} 총 {len(all_pairs)}개 (근거 충분 {n_grounded}개, 근거 불충분 {n_weak}개)")
        print(f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
