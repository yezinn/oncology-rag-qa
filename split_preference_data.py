"""
preference_data.jsonl을 DPO 학습 도구(mlx-lm-lora)가 기대하는 train/valid로 분리합니다.
시나리오(grounded/weak_evidence) 비율이 train/valid에 고르게 섞이도록 층화 분할합니다.

사용법:
    python3 split_preference_data.py
"""
import json
import os
import random

INPUT_PATH = "preference_data.jsonl"
OUTPUT_DIR = "data/dpo"
VALID_FRAC = 0.2  # 데이터가 25개뿐이라 20%만 떼어도 valid가 4~5개 정도

random.seed(42)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            # 학습 도구는 prompt/chosen/rejected 세 필드만 필요 -- 메타데이터(scenario 등)는 제외
            f.write(json.dumps(
                {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]},
                ensure_ascii=False,
            ) + "\n")


def split(rows, valid_frac):
    rows = rows[:]
    random.shuffle(rows)
    n_valid = max(1, round(len(rows) * valid_frac))
    return rows[n_valid:], rows[:n_valid]


def main():
    rows = load_jsonl(INPUT_PATH)
    grounded = [r for r in rows if r["scenario"] == "grounded"]
    weak = [r for r in rows if r["scenario"] == "weak_evidence"]

    g_train, g_valid = split(grounded, VALID_FRAC)
    w_train, w_valid = split(weak, VALID_FRAC)

    train = g_train + w_train
    valid = g_valid + w_valid
    random.shuffle(train)
    random.shuffle(valid)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(OUTPUT_DIR, "valid.jsonl"), valid)

    print(f"train={len(train)}개 (grounded {len(g_train)} + weak {len(w_train)})")
    print(f"valid={len(valid)}개 (grounded {len(g_valid)} + weak {len(w_valid)})")
    print(f"-> {OUTPUT_DIR}/train.jsonl, {OUTPUT_DIR}/valid.jsonl")


if __name__ == "__main__":
    main()
