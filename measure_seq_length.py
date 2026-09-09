"""
data/dpo/{train,valid,test}.jsonl의 prompt+chosen / prompt+rejected 토큰 길이를 실측해서
--max-seq-length를 감으로 정하지 않고 근거 있는 값으로 정하기 위한 진단 스크립트.

사용법:
    cd ~/Downloads/pubmed-rag-qa
    python3 measure_seq_length.py
"""
import json
import statistics

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
FILES = ["data/dpo/train.jsonl", "data/dpo/valid.jsonl", "data/dpo/test.jsonl"]


def load_tokenizer():
    # mlx_lm_lora 학습 시 실제로 쓰이는 것과 동일한 토크나이저 로더를 우선 시도.
    try:
        from mlx_lm.tokenizer_utils import load_tokenizer as _load
        from mlx_lm.utils import get_model_path

        model_path = get_model_path(MODEL)
        tok = _load(model_path)
        print(f"[tokenizer] mlx_lm.tokenizer_utils 사용 ({MODEL})")
        return tok
    except Exception as e:
        print(f"[tokenizer] mlx_lm 경로 실패 ({e!r}), transformers로 재시도")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"[tokenizer] transformers.AutoTokenizer 사용 ({MODEL})")
    return tok


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    tok = load_tokenizer()

    def tlen(text):
        return len(tok.encode(text))

    all_chosen_lens = []
    all_rejected_lens = []
    per_file_summary = []

    for path in FILES:
        try:
            rows = load_jsonl(path)
        except FileNotFoundError:
            print(f"  (건너뜀: {path} 없음)")
            continue

        chosen_lens = []
        rejected_lens = []
        for r in rows:
            prompt = r["prompt"]
            chosen_lens.append(tlen(prompt + r["chosen"]))
            rejected_lens.append(tlen(prompt + r["rejected"]))

        all_chosen_lens += chosen_lens
        all_rejected_lens += rejected_lens
        per_file_summary.append((path, len(rows), chosen_lens, rejected_lens))

    print("\n=== 파일별 요약 ===")
    for path, n, chosen_lens, rejected_lens in per_file_summary:
        combined = chosen_lens + rejected_lens
        combined.sort()
        print(f"{path} (n={n}, 토큰 길이 {len(combined)}개 = chosen+rejected)")
        print(f"  min={min(combined)} max={max(combined)} mean={statistics.mean(combined):.1f} "
              f"median={statistics.median(combined):.1f}")

    combined_all = sorted(all_chosen_lens + all_rejected_lens)
    n = len(combined_all)

    def pct(p):
        idx = min(n - 1, int(n * p))
        return combined_all[idx]

    print("\n=== 전체(train+valid+test, chosen+rejected 합산) 분포 ===")
    print(f"n={n}")
    print(f"min={combined_all[0]}  p50={pct(0.50)}  p90={pct(0.90)}  p95={pct(0.95)}  "
          f"p99={pct(0.99)}  max={combined_all[-1]}")

    recommended = pct(1.0)  # = max, 여유 위해 소폭 반올림
    recommended = ((recommended // 32) + 2) * 32  # 32의 배수로 올림 + 여유 1칸(32토큰)
    print(f"\n권장 --max-seq-length (실측 최댓값 기준, 32 단위 올림 + 여유): {recommended}")
    print("* 이 값보다 작게 잡으면 v5처럼 chosen/rejected를 가르는 답변 부분이 잘려나가")
    print("  loss가 0.693(ln2, '선호 없음' 기저값)으로 붕괴하는 예시가 다시 생길 수 있음.")


if __name__ == "__main__":
    main()
