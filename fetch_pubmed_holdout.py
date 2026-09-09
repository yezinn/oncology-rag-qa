"""
weak_evidence "경계선 예시" 재설계용 -- 기존 코퍼스와 같은 4개 도메인에서, 색인 안 된
다음 순위(151~180위) 논문을 별도로 수집합니다. 이 논문들은 data/abstracts.json(색인 대상)
에는 포함되지 않으므로 chroma_db에도 안 들어갑니다 -- 즉 "같은 도메인 어휘를 쓰지만 실제로는
검색이 안 되는" 진짜 근거 불충분 사례를 만드는 재료로 씁니다.

사전 준비:
    export ENTREZ_EMAIL="..."
    (fetch_pubmed.py를 먼저 실행해서 data/abstracts.json이 있어야 함 -- 중복 제외용)

사용법:
    python3 fetch_pubmed_holdout.py
"""
import os
import json
import time
from Bio import Entrez

Entrez.email = os.environ.get("ENTREZ_EMAIL", "your_email@example.com")

# fetch_pubmed.py와 동일한 4개 도메인 쿼리 (재현성을 위해 여기도 명시)
QUERIES = {
    "egfr_luad_prognosis": "EGFR mutation lung adenocarcinoma prognosis biomarker",
    "tnbc_chemo_response": "triple negative breast cancer neoadjuvant chemotherapy response prediction",
    "ssgsea_pathway": "ssGSEA pathway enrichment gene expression",
    "transfer_learning_drug_response": "transfer learning cell line patient drug response prediction",
}

RETSTART = 150  # 기존 코퍼스가 상위 150건을 색인했으므로 그다음 순위부터
RETMAX = 30     # 도메인당 30건씩, 총 120건


def search_pubmed(query, retstart, retmax):
    handle = Entrez.esearch(db="pubmed", term=query, retstart=retstart, retmax=retmax, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_abstracts(pmid_list):
    if not pmid_list:
        return []
    handle = Entrez.efetch(db="pubmed", id=",".join(pmid_list), rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    articles = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            pmid = str(medline["PMID"])
            article_data = medline["Article"]
            title = str(article_data.get("ArticleTitle", ""))
            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts)
            year = None
            try:
                year = str(article_data["Journal"]["JournalIssue"]["PubDate"].get("Year"))
            except (KeyError, TypeError):
                pass
            if abstract:
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                })
        except (KeyError, IndexError):
            continue
    return articles


def main():
    with open("data/abstracts.json", "r", encoding="utf-8") as f:
        indexed_pmids = {a["pmid"] for a in json.load(f)}

    all_articles = {}
    for topic, query in QUERIES.items():
        print(f"[{topic}] {RETSTART}~{RETSTART + RETMAX}위 검색 중: {query}")
        pmids = search_pubmed(query, RETSTART, RETMAX)
        print(f"  -> {len(pmids)}건 검색됨")
        articles = fetch_abstracts(pmids)
        for a in articles:
            if a["pmid"] in indexed_pmids:
                continue  # 혹시 겹치면 제외 (안전장치)
            a["topic"] = topic
            all_articles[a["pmid"]] = a
        time.sleep(0.4)

    result = list(all_articles.values())
    with open("data/abstracts_holdout.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(result)}건 수집 (색인된 코퍼스와 중복 없음)")
    print("-> data/abstracts_holdout.json")


if __name__ == "__main__":
    main()
