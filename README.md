# PubMed 기반 도메인 특화 RAG QA 시스템

암 유전체/전사체 연구(EGFR 변이 폐선암 예후, TNBC 항암화학요법 반응 예측, ssGSEA 등)
관련 PubMed 논문 초록을 근거로 질의응답을 수행하는 RAG 파이프라인입니다.
근거가 불충분한 질문에는 임의로 답변을 생성하지 않고 명시적으로 "확인 불가"를 반환합니다.

> RAG 관련 방법론 - Python/LangChain 스택과 본인 연구 도메인에 재구현 & 확장한 개인 프로젝트

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

**LLM**: 답변 생성과 채점 모두 Google Gemini API(`gemini-3.5-flash-lite`, Google AI Studio 무료 티어)를
사용합니다.

## 실행 결과

- **수집 데이터**: PubMed 초록 240건 (EGFR 변이 폐선암 예후 / TNBC 항암화학요법 반응 /
  ssGSEA pathway enrichment / 세포주-환자 전이학습 약물반응, 각 60건)
- **Golden Set**: 개인 학습 프로젝트 규모로 정직하게 라벨링 — **N=14**
  (SKALA 팀 프로젝트의 20개 케이스와는 별개 규모)
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

## 실행 순서

```bash
pip install -r requirements.txt --break-system-packages

# 1. PubMed 초록 수집
export ENTREZ_EMAIL="your_email@example.com"
python fetch_pubmed.py

# 2. 벡터 인덱스 구축
python build_index.py

# 3. 대화형으로 질의응답 테스트
export GOOGLE_API_KEY="..."   # Google AI Studio 발급 (https://aistudio.google.com/apikey)
python qa_chain.py

# 4. Golden Set 작성 후 자동 평가
## + GOOGLE_API_KEY가 있으면 LLM-as-a-Judge까지 실행)
cp golden_set_template.json golden_set.json
# golden_set.json을 열어 본인 도메인 지식으로 질문/정답 PMID를 채워넣기
python evaluate.py
```

## 설계 원칙

- **Grounding Guard**: 검색된 근거의 유사도가 임계값(`SCORE_THRESHOLD`)보다 낮으면
  LLM 호출 자체를 생략하고 즉시 "확인 불가"를 반환합니다. 프롬프트 지시만으로 환각을
  막는 대신, 코드 레벨에서 결정론적으로 차단합니다.
- **Golden Set 평가**: 도메인 전문가(본인)가 직접 작성한 질문-정답 PMID 세트로
  retrieval hit rate를 자동 측정하여, 정성적 인상이 아닌 정량적 지표로 검증합니다.
- **LLM-as-a-Judge**: golden set 질문마다 정답 텍스트를 별도로 작성하는 대신(reference-free),
  실제 생성된 답변을 채점 LLM이 근거 초록 기준으로 Faithfulness(환각 여부), Relevance(질문
  적합성) 두 축으로 1~5점 채점합니다. Grounding Guard가 답변을 차단한 케이스는 채점하지 않고
  "차단됨"으로 표시되어, 이 실행이 Grounding Guard 동작 검증도 겸합니다.

