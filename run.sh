#!/usr/bin/env bash
set -e

MODE=${1:-all}

python -m src.main   --mode "$MODE"   --dataset_name "Nan-Do/code-search-net-python"   --train_size 8000   --max_src_len 50   --max_tgt_len 80   --embed_dim 256   --hidden_dim 256   --batch_size 64   --epochs 12   --lr 3e-4   --teacher_forcing 0.5   --seed 42   --num_attention_examples 3
