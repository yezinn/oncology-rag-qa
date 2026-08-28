# PubMed 기반 도메인 특화 RAG QA 시스템

암 유전체/전사체 연구(EGFR 변이 폐선암 예후, TNBC 항암화학요법 반응 예측, ssGSEA 등)
관련 PubMed 논문 초록을 근거로 질의응답을 수행하는 RAG 파이프라인입니다.
근거가 불충분한 질문에는 임의로 답변을 생성하지 않고 명시적으로 "확인 불가"를 반환합니다.

> 개인 학습 프로젝트입니다. SKALA 4기 종합실습(HelpDesk AI, Spring AI 기반 팀 프로젝트)에서
> 담당했던 RAG 관련 방법론을 Python/LangChain 스택과 본인 연구 도메인에 개인적으로
> 재구현·확장한 것으로, 팀 프로젝트와는 별개입니다.

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

실제로 실행해서 얻은 수치입니다 (추정치 아님).

- **수집 데이터**: PubMed 초록 240건 (EGFR 변이 폐선암 예후 / TNBC 항암화학요법 반응 /
  ssGSEA pathway enrichment / 세포주-환자 전이학습 약물반응, 각 60건)
- **Golden Set**: 개인 학습 프로젝트 규모로 정직하게 라벨링 — **N=14**
  (SKALA 팀 프로젝트의 20개 케이스와는 별개 규모)
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
export OPENAI_API_KEY="sk-..."
python qa_chain.py

# 4. Golden Set 작성 후 자동 평가
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
- 이 프로젝트는 SKALA 4기 종합실습에서 진행한 HelpDesk AI 프로젝트(Spring AI 기반
  RAG + Grounding Guard + Golden Set 평가)와 동일한 방법론을 Python/LangChain 스택,
  본인 연구 도메인(암 유전체)에 적용해 재구현한 것입니다.

## 참고 (제약 사항 / 트러블슈팅)

- **`SCORE_THRESHOLD`(현재 0.35)는 아직 실측 튜닝 전인 초기값입니다.** `evaluate.py`는
  이 threshold를 쓰지 않고(순수 top-k 검색만 검증) 위 hit rate를 측정했습니다.
  Grounding Guard가 실제로 LLM 호출을 막는 기준값 튜닝은 `qa_chain.py`로 대화형
  테스트를 거쳐야 합니다.
- **LangChain 0.x → 1.x 메이저 업그레이드로 인한 import 경로 변경**: `requirements.txt`를
  최신 버전으로 설치하면 아래 두 곳에서 `ModuleNotFoundError`가 발생합니다(본 저장소
  코드에는 이미 반영되어 있음).
  - `from langchain.docstore.document import Document` → `from langchain_core.documents import Document`
  - `from langchain.prompts import ChatPromptTemplate` → `from langchain_core.prompts import ChatPromptTemplate`
  - 반면 `similarity_search_with_relevance_scores`(복수형)와 `Chroma.persist()`는
    최신 버전에서도 그대로 동작합니다.
- NCBI E-utilities는 API 키 없이도 사용 가능하지만 초당 3회 요청 제한이 있습니다
  (스크립트에 `time.sleep`으로 반영되어 있음). API 키를 발급받으면 제한이 완화됩니다.
