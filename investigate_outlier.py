"""
measure_seq_length.py에서 발견된 이상치(max=28030, p99=3732와 큰 차이)의 정체를 확인.
어떤 example이 튀는지, prompt 안에 컨텍스트가 중복/과다 삽입된 건 아닌지 점검.
"""
import json
from transformers import AutoTokenizer

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
FILES = ["data/dpo/train.jsonl", "data/dpo/valid.jsonl", "data/dpo/test.jsonl"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)

    records = []
    for path in FILES:
        try:
            rows = load_jsonl(path)
        except FileNotFoundError:
            continue
        for i, r in enumerate(rows):
            prompt = r["prompt"]
            plen = len(tok.encode(prompt))
            clen = len(tok.encode(prompt + r["chosen"]))
            rlen = len(tok.encode(prompt + r["rejected"]))
            records.append({
                "file": path, "idx": i, "scenario": r.get("scenario"),
                "source_pmid": r.get("source_pmid"),
                "question": r.get("question", r["prompt"])[:80],
                "prompt_chars": len(prompt), "prompt_tokens": plen,
                "chosen_total_tokens": clen, "rejected_total_tokens": rlen,
            })

    records.sort(key=lambda x: max(x["chosen_total_tokens"], x["rejected_total_tokens"]), reverse=True)

    print("=== 토큰 길이 상위 10개 ===")
    for rec in records[:10]:
        print(rec)

    top = records[0]
    print(f"\n=== 최상위 1건 상세 (파일={top['file']}, idx={top['idx']}) ===")

    # 해당 example의 실제 prompt 텍스트를 다시 읽어서 구조를 확인
    rows = load_jsonl(top["file"])
    r = rows[top["idx"]]
    prompt = r["prompt"]
    print(f"prompt 총 글자수: {len(prompt)}")
    print(f"scenario: {r.get('scenario')}, source_pmid: {r.get('source_pmid')}")

    # [검색된 논문 초록] 구획이 몇 번 나오는지 등으로 중복 삽입 여부 체크
    marker_count = prompt.count("PMID")
    print(f"prompt 내 'PMID' 등장 횟수 (컨텍스트에 포함된 문서 수 추정): {marker_count}")

    # 프롬프트 앞부분과 끝부분만 출력해서 구조 확인 (전체 덤프는 생략)
    print("\n--- prompt 앞 500자 ---")
    print(prompt[:500])
    print("\n--- prompt 뒤 500자 ---")
    print(prompt[-500:])


if __name__ == "__main__":
    main()
