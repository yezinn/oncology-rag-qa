"""
mlx_lm_lora.train을 CLI로 직접 부르는 대신 이 스크립트로 감싸서 실행.
MLX는 스텝마다 GPU 메모리를 완전히 반납하지 않고 캐시로 쌓아두는 특성이 있어서,
길이가 들쭉날쭉한 학습 데이터를 여러 스텝 연달아 돌리면 캐시가 누적되어
결국 wired_limit을 넘겨 OOM이 나는 것으로 추정됨 (긴 예시 하나만으로는 문제 없었음
-- 스트레스 테스트로 확인).
mx.set_cache_limit()으로 캐시 상한을 낮게 잡아 매 스텝 적극적으로 메모리를
반납하도록 강제 -> 속도는 조금 느려지지만 메모리 누적을 막음.

사용법:
    cd ~/Downloads/pubmed-rag-qa
    python3 run_train_v6.py
"""
import sys
import mlx.core as mx

CACHE_LIMIT_BYTES = 512 * 1024 * 1024  # 512MB -- 누적을 막기 위해 낮게 설정
mx.set_cache_limit(CACHE_LIMIT_BYTES)
print(f"[run_train_v6] mx.set_cache_limit({CACHE_LIMIT_BYTES} bytes = 512MB) 적용")

sys.argv = [
    "mlx_lm_lora.train",
    "--model", "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "--train",
    "--data", "data/dpo",
    "--train-type", "lora",
    "--train-mode", "dpo",
    "--num-layers", "8",
    "--batch-size", "1",
    "--grad-checkpoint",
    "--val-batches", "2",
    "--test-batches", "2",
    "--iters", "485",
    "--adapter-path", "adapters_v6",
    "--test",
    "--fuse",
]

from mlx_lm_lora.train import main

main()
