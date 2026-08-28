# PubMed 기반 도메인 특화 RAG QA 시스템

암 유전체/전사체 연구(EGFR 변이 폐선암 예후, TNBC 항암화학요법 반응 예측, ssGSEA 등)
관련 PubMed 논문 초록을 근거로 질의응답을 수행하는 RAG 파이프라인
근거가 불충분한 질문에는 임의로 답변을 생성하지 않고 명시적으로 "확인 불가"를 반환한다.

## 아키텍처

```
PubMed E-utilities API (fetch_pubmed.py)
        │
        ▼
초록 임베딩 → Chroma 벡터 스토어 (build_index.py)
        │
        ▼
RAG QA + Grounding Guard (qa_chain.py)
        │
        ▼
Golden Set 기반 자동 평가 + LLM-as-a-Judge (evaluate.py)
```

**LLM**: 답변 생성과 채점 모두 Google Gemini API(`gemini-3.5-flash-lite`, Google AI Studio)를
사용한다.

## 실행 결과

- **수집 데이터**: PubMed 초록 240건 (EGFR 변이 폐선암 예후 / TNBC 항암화학요법 반응 /
  ssGSEA pathway enrichment / 세포주-환자 전이학습 약물반응, 각 60건)
- **Golden Set**: **N=14**
- **Retrieval Hit Rate: 78.6% (11/14)** — top-5 검색 결과에 정답 논문(PMID)이 포함된 비율.
  잔여 실패 3건은 질문이 구체적인데도 근거 논문이 top-5 밖으로 밀린 경우로, 임베딩 유사도
  랭킹 자체의 한계로 판단.
- **Grounding Guard 통과율: 57.1% (8/14, threshold=0.30)** — top-1 유사도 점수가 임계값 미만이면
  LLM 호출 없이 즉시 차단. threshold는 `check_scores.py`로 실측한 hit/miss 점수 분포 기반으로
  0.35 -> 0.30으로 조정 (오답 통과율은 1/3로 동일하게 유지하면서 정답 통과율만 6/11 -> 7/11로
  개선).
- **LLM-as-a-Judge (reference-free)**: Grounding Guard를 통과해 실제 답변이 생성된 8건 전량
  채점 — 평균 Faithfulness 5.0/5, 평균 Relevance 5.0/5. Guard가 통과시킨 답변은 전량 근거에
  충실하고 질문에 적합했음을 확인 (차단된 6건은 애초에 LLM을 호출하지 않아 채점 대상에서
  제외). 자세한 사례는 `evaluation_result.json` 참고.

## sLLM 확장: DPO 파인튜닝 실험

기존 RAG 파이프라인과 Golden Set을 재사용해, 로컬 sLLM(`mlx-community/Qwen2.5-3B-Instruct-4bit`)에
DPO(LoRA, Apple MLX) 적용해본다. Grounding Guard의 판단(근거 충분 시 인용 답변 / 근거
불충분 시 거절)을 모델 자체에 내재화할 수 있는지가 목적이다.

**Preference 데이터**: 근거 충분/불충분 두 시나리오로 25쌍(13 + 12)을 프로그램적으로 구성.
Golden Set 14개가 정답으로 쓰는 PMID는 학습 데이터에서 전부 제외해 train/eval을 분리함
(`build_preference_data.py`, `split_preference_data.py`).

**학습**: LoRA(rank 8, 8개 레이어), 100 iters, `mlx-lm-lora`. Train loss 0.693 → 0.003,
val loss 0.693 → 0.034로 빠르게 수렴 (train 20개뿐이라 과적합 가능성을 염두에 두고 아래
held-out 평가로 별도 검증).

**Held-out 평가 (Golden Set 14개, 학습에 전혀 쓰이지 않은 질문 — `evaluate_sllm.py`)**:

| 지표 | Base (파인튜닝 전) | DPO 파인튜닝 후 |
|---|---|---|
| Citation Rate (답변 중 PMID 인용 비율) | 50.0% (7/14) | 64.3% (9/14) |
| Self-Refusal Accuracy (모델 스스로의 거절 판단이 Guard의 threshold 판단과 일치하는 비율) | 57.1% | 57.1% (동일) |

**결과 해석**: Citation Rate는 실제로 개선됐지만, Self-Refusal Accuracy는 두 모델이 완전히 같은
값을 가짐 — 그 이유는 두 모델 다 14문항 중 단 한 번도 스스로 답변을 거절하지 않았기 때문이다.
57.1%라는 수치는 Golden Set에서 원래 Guard가 차단하는 문항 비율(8/14)과 우연히 일치해서 나온
값일 뿐, 실제 판단 능력의 개선을 의미하지 않는다. 즉 이번 규모(학습 pair 20개)의 DPO는
**표층적 출력 스타일(인용 형식)은 개선했지만, "근거가 충분한가"라는 더 깊은 판단은 일반화하지
못했다** — 코드 레벨 Grounding Guard가 여전히 필수적임을 실측으로 확인한 결과이다.

한계: preference pair 25개는 통계적으로 작은 규모이며, 학습 도구가 요구하는 `test.jsonl`은
데이터 부족으로 `valid.jsonl`을 그대로 재사용함(`evaluate_sllm.py`의
Golden Set 평가만이 실제 held-out 검증).

## 실행 순서

```bash
pip install -r requirements.txt --break-system-packages

# 1. PubMed 초록 수집
export ENTREZ_EMAIL="your_email@example.com"
python fetch_pubmed.py

# 2. 벡터 인덱스 구축
python build_index.py

# 3. 대화형 질의응답 테스트
export GOOGLE_API_KEY="..."   # Google AI Studio 발급 (https://aistudio.google.com/apikey)
python qa_chain.py

# 4. Golden Set 작성 후 자동 평가
## + GOOGLE_API_KEY가 있으면 LLM-as-a-Judge까지 실행
cp golden_set_template.json golden_set.json
# golden_set.json을 열어 본인 도메인 지식으로 질문/정답 PMID를 채워넣기
python evaluate.py

# + sLLM DPO 확장 -- preference 데이터 구축 + train/valid 분리
python build_preference_data.py
python split_preference_data.py

# + DPO 파인튜닝 (Apple MLX, mlx-lm-lora 필요)
pip install -U mlx-lm mlx-lm-lora --break-system-packages
mlx_lm_lora.train \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --train --train-type lora --train-mode dpo \
  --data data/dpo --beta 0.1 --dpo-cpo-loss-type sigmoid \
  --reference-model-path mlx-community/Qwen2.5-3B-Instruct-4bit \
  --num-layers 8 --batch-size 1 --iters 100 --learning-rate 5e-6 \
  --gradient-accumulation-steps 4 --val-batches -1 --grad-checkpoint

# + sLLM 평가 (Golden Set held-out 비교)
python evaluate_sllm.py
```

## 설계 원칙

- **Grounding Guard**: 검색된 근거의 유사도가 임계값(`SCORE_THRESHOLD`)보다 낮으면
  LLM 호출 자체를 생략하고 즉시 "확인 불가"를 반환한다. 프롬프트 지시만으로 환각을
  막는 대신, 코드 레벨에서 결정론적으로 차단한다.
- **Golden Set 평가**: 도메인 전문가(본인)가 직접 작성한 질문-정답 PMID 세트로
  retrieval hit rate를 자동 측정하여, 정성적 인상이 아닌 정량적 지표로 검증한다.
- **LLM-as-a-Judge**: golden set 질문마다 정답 텍스트를 별도로 작성하는 대신(reference-free),
  실제 생성된 답변을 채점 LLM이 근거 초록 기준으로 Faithfulness(환각 여부), Relevance(질문
  적합성) 두 축으로 1~5점 채점. Grounding Guard가 답변을 차단한 케이스는 채점하지 않고
  "차단됨"으로 표시되어, 이 실행이 Grounding Guard 동작 검증도 겸한다.
- **DPO로 보완**: 소규모 preference 데이터로 로컬 sLLM에 DPO를 적용한 결과,
  출력 스타일은 개선됐지만 근거 충분성 판단 자체는 일반화되지 않았다. 즉 학습된
  성향(alignment)만으로는 안전성을 보장할 수 없고, 코드 레벨의 결정론적 Guard가 여전히
  필요하다는 것을 실측으로 확인했다.

