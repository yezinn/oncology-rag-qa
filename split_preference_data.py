"""
preference_data.jsonl을 DPO 학습 도구(mlx-lm-lora)가 기대하는 train/valid/test로 분리합니다.
시나리오(grounded/weak_evidence) 비율이 세 세트에 고르게 섞이도록 층화 분할합니다.

주의: 데이터가 25개뿐이던 초기 버전에서는 test.jsonl을 valid.jsonl 복사본으로 대신했으나
(진짜 held-out test set이 아니었음), 코퍼스/preference pair 규모를 확장한 뒤로는 train/valid/test를
실제로 3분할합니다.

사용법:
    python3 split_preference_data.py
"""
import json
import os
import random

INPUT_PATH = "preference_data.jsonl"
OUTPUT_DIR = "data/dpo"
VALID_FRAC = 0.15
TEST_FRAC = 0.15

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


def split3(rows, valid_frac, test_frac):
    rows = rows[:]
    random.shuffle(rows)
    n = len(rows)
    n_valid = max(1, round(n * valid_frac))
    n_test = max(1, round(n * test_frac))
    test = rows[:n_test]
    valid = rows[n_test:n_test + n_valid]
    train = rows[n_test + n_valid:]
    return train, valid, test


def main():
    rows = load_jsonl(INPUT_PATH)
    grounded = [r for r in rows if r["scenario"] == "grounded"]
    weak = [r for r in rows if r["scenario"] == "weak_evidence"]

    g_train, g_valid, g_test = split3(grounded, VALID_FRAC, TEST_FRAC)
    w_train, w_valid, w_test = split3(weak, VALID_FRAC, TEST_FRAC)

    train = g_train + w_train
    valid = g_valid + w_valid
    test = g_test + w_test
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(OUTPUT_DIR, "valid.jsonl"), valid)
    write_jsonl(os.path.join(OUTPUT_DIR, "test.jsonl"), test)

    print(f"train={len(train)}개 (grounded {len(g_train)} + weak {len(w_train)})")
    print(f"valid={len(valid)}개 (grounded {len(g_valid)} + weak {len(w_valid)})")
    print(f"test={len(test)}개 (grounded {len(g_test)} + weak {len(w_test)})")
    print(f"-> {OUTPUT_DIR}/train.jsonl, {OUTPUT_DIR}/valid.jsonl, {OUTPUT_DIR}/test.jsonl")


if __name__ == "__main__":
    main()
