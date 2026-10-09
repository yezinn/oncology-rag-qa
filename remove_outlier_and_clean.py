"""
PMID 27885969(학회 초록집, 8만자 이상치)를 코퍼스에서 제거하고,
이 문서를 컨텍스트로 사용한 preference_data.jsonl 항목도 함께 제거.
"""
import json
import shutil

BAD_PMID = "27885969"

# 1) abstracts.json에서 제거 (백업 먼저)
shutil.copy("data/abstracts.json", "data/abstracts_before_cleanup.json")
with open("data/abstracts.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)

before = len(abstracts)
abstracts = [a for a in abstracts if str(a.get("pmid")) != BAD_PMID]
after = len(abstracts)
print(f"abstracts.json: {before}건 -> {after}건 (PMID {BAD_PMID} 제거, 백업: data/abstracts_before_cleanup.json)")

with open("data/abstracts.json", "w", encoding="utf-8") as f:
    json.dump(abstracts, f, ensure_ascii=False, indent=2)

# 2) preference_data.jsonl에서 이 문서를 컨텍스트로 쓴 항목 제거
shutil.copy("preference_data.jsonl", "preference_data_before_cleanup.jsonl")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


rows = load_jsonl("preference_data.jsonl")
kept = [r for r in rows if f"PMID: {BAD_PMID}" not in r["prompt"]]
removed_n = len(rows) - len(kept)
print(f"preference_data.jsonl: {len(rows)}개 -> {len(kept)}개 ({removed_n}개 제거, 백업: preference_data_before_cleanup.jsonl)")

with open("preference_data.jsonl", "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\n다음 순서로 진행하면 됨:")
print("1. rm -rf chroma_db && python3 build_index.py   # 코퍼스 재색인 (API 불필요, 로컬)")
print("2. python3 split_preference_data.py              # train/valid/test 재분리")
print("3. python3 measure_seq_length.py                 # 정제된 데이터로 max-seq-length 재측정")
