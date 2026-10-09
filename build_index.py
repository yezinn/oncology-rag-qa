"""
수집한 PubMed 초록을 임베딩하여 로컬 벡터 스토어(Chroma)를 구축합니다.
sentence-transformers를 사용하므로 임베딩 자체에는 API 비용이 들지 않습니다.

사전 준비:
    pip install langchain langchain-community chromadb sentence-transformers --break-system-packages

사용법:
    python build_index.py

주의: SCORE_THRESHOLD 관련 버그(negative relevance score)를 고치면서 임베딩
정규화 방식이 바뀌었습니다. 이전에 만든 ./chroma_db가 있다면 삭제하고
다시 실행해야 새 설정으로 인덱스가 재구축됩니다.

2026-10-08 임베딩 모델 교체: all-MiniLM-L6-v2(영어 전용) -> paraphrase-multilingual-MiniLM-L12-v2
(다국어). 원인: golden set 질문이 한국어인데 영어 전용 모델을 쓰다 보니, 정답 문서의
제목/초록 내용이 질문과 명백히 일치해도 dense 검색이 아예 못 찾는 cross-lingual
불일치가 실측으로 확인됨(같은 질문을 영어로 번역만 해도 순위가 top-50 밖 -> 1위로
뛰어오름 — 코퍼스나 로직 문제가 아니라 임베딩 모델의 언어 한계였음). 이 모델 교체는
**반드시 ./chroma_db를 삭제하고 재실행**해야 함 — 차원은 둘 다 384로 같지만 벡터
공간 자체가 다른 모델이라 기존 인덱스와 호환 안 됨. 재실행 후 qa_chain.py의
SCORE_THRESHOLD도 check_scores.py로 반드시 재튜닝해야 함(점수 분포가 달라짐).
"""
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_PATH = "data/abstracts.json"
PERSIST_DIR = "./chroma_db"
# 다국어(한국어 포함) 지원 모델 — all-MiniLM-L6-v2와 동일한 384차원 출력이라
# 나머지 파이프라인(Chroma cosine space 설정 등)은 그대로 호환됨. E5 계열과 달리
# "query: "/"passage: " 프리픽스가 필요 없어 코드 변경이 최소화됨.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_documents():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        abstracts = json.load(f)

    docs = []
    for a in abstracts:
        content = f"Title: {a['title']}\n\nAbstract: {a['abstract']}"
        docs.append(Document(
            page_content=content,
            metadata={
                "pmid": a["pmid"],
                "title": a["title"],
                "year": a.get("year", "") or "",
                "topic": a.get("topic", ""),
            },
        ))
    return docs


def get_embeddings():
    # normalize_embeddings=True로 단위벡터화 + Chroma를 cosine distance로 구성해야
    # similarity_search_with_relevance_scores()가 0~1 범위의 의미 있는 점수를 반환합니다.
    # (정규화 없이 기본 L2 거리로 두면 relevance score가 음수로 튀는 버그가 있었음 —
    #  qa_chain.py의 get_vectorstore()도 반드시 같은 설정을 써야 함)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def main():
    docs = load_documents()
    print(f"{len(docs)}개 문서 로드 완료")

    # M5 MacBook: HuggingFaceEmbeddings는 기본적으로 CPU를 쓰지만 문서 수가
    # 적어(수백 건) 속도에는 큰 문제가 없습니다.
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vectorstore.persist()
    print(f"벡터 스토어 구축 완료 -> {PERSIST_DIR}")


if __name__ == "__main__":
    main()
