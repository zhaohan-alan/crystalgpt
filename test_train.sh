#!/bin/bash

python main.py \
    --batchsize 90 \
    --epochs 2000 \
    --train_path data/train_comp_cleaned.csv \
    --valid_path data/val_comp_cleaned.csv \
    --test_path data/test_comp_cleaned.csv \
    --folder ./test_output/ \
    --num_io_process 40 \
    --model_size 32 \
    --dropout_rate 0.4 \
    --attn_dropout 0.4 \
    --val_interval 100 \
    --Kl 16 \
    --transformer_layers 16 \
    --num_heads 8 \
    --use_comp_feature \
    --comp_feature_dim 256 \
    --lr 5e-4 \
    | tee test_train.log 