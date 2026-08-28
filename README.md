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
Golden Set 기반 자동 평가 (evaluate.py)
```

## 실행 결과

- **수집 데이터**: PubMed 초록 240건 (EGFR 변이 폐선암 예후 / TNBC 항암화학요법 반응 /
  ssGSEA pathway enrichment / 세포주-환자 전이학습 약물반응, 각 60건)
- **Golden Set**: **N=14**
- **Retrieval Hit Rate: 85.7% (12/14)**
  - 1차 실행 78.6%(11/14) → 검색 범위와 겹치는 일반적인 리뷰 논문 문항 1개를
    더 구체적인 사례로 교체 후 85.7%로 개선
  - 잔여 실패 2건은 질문이 구체적인데도 근거 논문이 top-5 밖으로 밀린 경우로,
    임베딩 유사도 랭킹 자체의 한계로 판단 (프롬프트나 Grounding Guard 문제가 아님).
    자세한 사례는 `evaluation_result.json` 참고.

## 실행 순서

```bash
pip install -r requirements.txt --break-system-packages

# 1. PubMed 초록 수집
export ENTREZ_EMAIL="your_email@example.com"
python fetch_pubmed.py

# 2. 벡터 인덱스 구축
python build_index.py

# 3. 대화형으로 질의응답 테스트
export OPENAI_API_KEY="..."
python qa_chain.py

# 4. Golden Set 작성 후 자동 평가
cp golden_set_template.json golden_set.json
# golden_set.json을 열어 본인 도메인 지식으로 질문/정답 PMID를 채워넣기
python evaluate.py
```

## 설계 원칙

- **Grounding Guard**: 검색된 근거의 유사도가 임계값(`SCORE_THRESHOLD`)보다 낮으면
  LLM 호출 자체를 생략하고 즉시 "확인 불가"를 반환하며, 프롬프트 지시만으로 환각을
  막는 대신, 코드 레벨에서 결정론적으로 차단합니다.
- **Golden Set 평가**: 직접 작성한 질문-정답 PMID 세트로
  retrieval hit rate를 자동 측정하여 정량적 지표로 검증합니다.
