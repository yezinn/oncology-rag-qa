"""
prompt+chosen / prompt+rejected 토큰 길이가 THRESHOLD를 넘는 preference pair를
아예 제외. --max-seq-length 플래그 대신 데이터 자체를 정제해서, 학습 시엔
플래그 없이 (v5와 동일하게, 메모리 안전성 검증된 방식으로) 돌리기 위함.
"""
import json
import shutil
from transformers import AutoTokenizer

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
THRESHOLD = 3072
INPUT_PATH = "preference_data.jsonl"


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
    shutil.copy(INPUT_PATH, "preference_data_before_length_filter.jsonl")
    rows = load_jsonl(INPUT_PATH)

    kept, dropped = [], []
    for r in rows:
        prompt = r["prompt"]
        clen = len(tok.encode(prompt + r["chosen"]))
        rlen = len(tok.encode(prompt + r["rejected"]))
        if max(clen, rlen) <= THRESHOLD:
            kept.append(r)
        else:
            dropped.append((r.get("scenario", "grounded"), max(clen, rlen)))

    print(f"threshold={THRESHOLD} 토큰 기준")
    print(f"{len(rows)}개 -> 유지 {len(kept)}개 / 제외 {len(dropped)}개")
    for scenario, length in dropped:
        print(f"  제외: scenario={scenario} length={length}")

    with open(INPUT_PATH, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{INPUT_PATH} 갱신 완료 (백업: preference_data_before_length_filter.jsonl)")


if __name__ == "__main__":
    main()
