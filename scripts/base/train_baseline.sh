#!/bin/bash

source scripts/configs/common.sh

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
  --save_dir "checkpoints/base_runs/${MODEL}_${SEED}" \
  --save_every 1000 \
  --name "${MODEL}_${SEED}_base"