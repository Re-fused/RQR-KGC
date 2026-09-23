#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
DATA_PATH=${1:-data/wn18rr}
OUTPUT_PATH=${2:-outputs/wn18rr}
GPU_ID=${3:-0}
CUDA_VISIBLE_DEVICES="$GPU_ID" python -u code/train.py \
  --data_path "$DATA_PATH" --save_path "$OUTPUT_PATH" --dataset wn18rr \
  --do_train --do_valid --do_test --cuda \
  --negative_sample_size 128 --batch_size 2048 --hidden_dim 1000 --gamma 12 \
  --test_batch_size 8 --learning_rate 0.001 --residual_lr_scale 0.5 \
  --scale_lr_scale 2 --max_steps 100000 --decay_steps 20000 50000 80000 \
  --regularization 0.4 --residual_regularization 0.0001 \
  --adversarial_temperature 0.5 --rank_loss_weight 0.025 \
  --rank_loss_margin 0.5 --rank_loss_topk 10 --rank_loss_cutoffs 1 3 10 \
  --curriculum_hard_negative_min_ratio 0.05 \
  --curriculum_hard_negative_max_ratio 0.15 \
  --curriculum_hard_negative_start_step 4000 \
  --curriculum_hard_negative_ramp_steps 8000 \
  --valid_steps 2000 --log_steps 100 --cpu_num 8 --seed 20260806
