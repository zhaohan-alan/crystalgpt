#!/bin/bash

# 使用组成特征的CrystalFormer训练脚本
# 基于原有的train.sh，但使用带组成特征的数据文件

python main.py \
    --batchsize 80 \
    --epochs 2000 \
    --train_path data/train_comp_cleaned.csv \
    --valid_path data/val_comp_cleaned.csv \
    --test_path data/test_comp_cleaned.csv \
    --folder ./data/ \
    --num_io_process 40 \
    --model_size 32 \
    --dropout_rate 0.5 \
    --attn_dropout 0.5 \
    --val_interval 50 \
    --Kl 16 | tee train_comp.log 