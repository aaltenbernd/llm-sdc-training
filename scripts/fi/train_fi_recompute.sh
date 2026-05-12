#!/bin/bash

source scripts/configs/common.sh
source scripts/configs/nvbit_default.sh

python torchrun_main.py \
  --single_gpu \
  --project "$PROJECT" \
  --seed "$SEED" \
  --model_config "configs/${MODEL}.json" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --total_batch_size 512 \
  --num_training_steps 100000 \
  --eval_every "$EVAL_EVERY" \
  --exit_after "$EXIT_AFTER" \
  --grad_clipping 1.0 \
  --warmup_steps 1000 \
  --weight_decay 0.01 \
  --dtype bfloat16 \
  --optimizer adamw \
  --base_model_path "checkpoints/base_runs/${MODEL}_${SEED}_${LR}" \
  --compare_every 1000 \
  --name "${MODEL}_${SEED}_fi_recompute" \
  --fi_nvbit_enable \
  --fi_nvbit_location backward \
  --fi_nvbit_trigger_rate 100 \
  --fi_nvbit_recompute \
  --fi_nvbit_alpha 0.05 \
  --fi_nvbit_target_funcs 0 1 2 3 4 5 6 7 8 9 \
  --fi_nvbit_duration 5 \
  --fi_nvbit_duration_random \
  --fi_nvbit_steps -1
