"""
data/abstracts.json에서 비정상적으로 긴 초록(학회 초록집 등 단일 논문이 아닌 항목)을 스캔.
"""
import json
import statistics

with open("data/abstracts.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)

lens = [(len(a.get("abstract", "")), a) for a in abstracts]
lens.sort(key=lambda x: x[0], reverse=True)

char_lens = [l for l, _ in lens]
print(f"총 {len(abstracts)}건")
print(f"글자수 분포: min={min(char_lens)} median={statistics.median(char_lens):.0f} "
      f"mean={statistics.mean(char_lens):.0f} p95={char_lens[int(len(char_lens)*0.05)]} "
      f"max={max(char_lens)}")

print("\n=== 글자수 상위 15건 ===")
for l, a in lens[:15]:
    print(f"{l:>7}자  PMID={a.get('pmid')}  topic={a.get('topic')}  title={a.get('title','')[:70]}")

THRESHOLD = 3000
outliers = [a for l, a in lens if l > THRESHOLD]
print(f"\n{THRESHOLD}자 초과 항목: {len(outliers)}건")
for a in outliers:
    print(f"  PMID={a.get('pmid')} topic={a.get('topic')} len={len(a.get('abstract',''))}")
