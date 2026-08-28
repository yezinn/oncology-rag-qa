"""
수집한 PubMed 초록을 임베딩하여 로컬 벡터 스토어(Chroma)를 구축합니다.
sentence-transformers를 사용하므로 임베딩 자체에는 OpenAI API 비용이 들지 않습니다.

사전 준비:
    pip install langchain langchain-community chromadb sentence-transformers --break-system-packages

사용법:
    python build_index.py
"""
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_PATH = "data/abstracts.json"
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


def main():
    docs = load_documents()
    print(f"{len(docs)}개 문서 로드 완료")

    # M5 MacBook: HuggingFaceEmbeddings는 기본적으로 CPU를 쓰지만 문서 수가
    # 적어(수백 건) 속도에는 큰 문제가 없습니다.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectorstore.persist()
    print(f"벡터 스토어 구축 완료 -> {PERSIST_DIR}")


if __name__ == "__main__":
    main()
