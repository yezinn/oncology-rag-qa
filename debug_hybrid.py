"""
Hybrid Retriever 적용 후에도 Hit Rate가 그대로인 이유를 진단하는 스크립트.
BM25 단독 / Dense 단독 결과를 분리해서 보여줘, Hit Rate 측정에서 miss로 나온
케이스들에서 BM25가 애초에 정답 PMID를 찾아내는지 확인한다.
(API 키 필요 없음 — 로컬 임베딩/BM25만 사용)

사용법:
    python debug_hybrid.py
"""
import json
from qa_chain import get_vectorstore, TOP_K
from langchain_community.retrievers import BM25Retriever
from build_index import load_documents

GOLDEN_SET_PATH = "golden_set.json"
# evaluate.py 실행 결과에서 hit=False로 나온 케이스. 바뀌면 여기만 수정.
TARGET_IDS = {"q-001", "q-005", "q-008"}


def main():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    vectorstore = get_vectorstore()
    docs = load_documents()
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = TOP_K

    for case in golden_set:
        if case["id"] not in TARGET_IDS:
            continue
        expected = set(case["expected_pmids"])

        bm25_docs = bm25.invoke(case["question"])
        bm25_pmids = [d.metadata.get("pmid") for d in bm25_docs]
        bm25_hit = bool(expected & set(bm25_pmids))

        dense_docs = vectorstore.similarity_search(case["question"], k=TOP_K)
        dense_pmids = [d.metadata.get("pmid") for d in dense_docs]
        dense_hit = bool(expected & set(dense_pmids))

        print(f"[{case['id']}] 질문: {case['question']}")
        print(f"  기대 PMID: {sorted(expected)}")
        print(f"  BM25 단독  top{TOP_K}: {bm25_pmids}  -> hit={bm25_hit}")
        print(f"  Dense 단독 top{TOP_K}: {dense_pmids}  -> hit={dense_hit}")
        print()


if __name__ == "__main__":
    main()
