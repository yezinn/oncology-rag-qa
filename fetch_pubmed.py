"""
PubMed 초록 수집 스크립트
EGFR 변이 폐선암, TNBC 항암화학요법 반응, ssGSEA 등 연구 도메인 관련 논문 초록을 수집합니다.

사전 준비:
    pip install biopython --break-system-packages
    export ENTREZ_EMAIL="your_email@example.com"

사용법:
    python fetch_pubmed.py
"""
import os
import json
import time
from Bio import Entrez

Entrez.email = os.environ.get("ENTREZ_EMAIL", "your_email@example.com")

# 본인 연구 도메인에 맞게 쿼리를 자유롭게 수정하세요.
QUERIES = {
    "egfr_luad_prognosis": "EGFR mutation lung adenocarcinoma prognosis biomarker",
    "tnbc_chemo_response": "triple negative breast cancer neoadjuvant chemotherapy response prediction",
    "ssgsea_pathway": "ssGSEA pathway enrichment gene expression",
    "transfer_learning_drug_response": "transfer learning cell line patient drug response prediction",
}

MAX_RESULTS_PER_QUERY = 60


def search_pubmed(query, max_results=MAX_RESULTS_PER_QUERY):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
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
    all_articles = {}
    for topic, query in QUERIES.items():
        print(f"[{topic}] 검색 중: {query}")
        pmids = search_pubmed(query)
        print(f"  -> {len(pmids)}건 검색됨")
        articles = fetch_abstracts(pmids)
        for a in articles:
            a["topic"] = topic
            all_articles[a["pmid"]] = a  # PMID 기준 중복 제거
        time.sleep(0.4)  # NCBI rate limit 준수 (초당 3회 이하)

    os.makedirs("data", exist_ok=True)
    out_path = "data/abstracts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(all_articles.values()), f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(all_articles)}개 초록을 {out_path}에 저장했습니다.")


if __name__ == "__main__":
    main()
