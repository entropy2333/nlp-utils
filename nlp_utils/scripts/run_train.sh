#!/bin/bash
set -x
set -e
if [ ! -n "$1" ]; then
	echo "Please input tag."
	exit 0
fi
TAG=$1
DATE=`date +%Y%m%d_%H%M%S`
export CUDA_VISIBLE_DEVICES=0
nohup python -u \
    main.py \
    --bert_dir hfl/chinese-roberta-wwm-ext \
    --save_model_path save/${DATE}_${TAG} \
    --max_epochs 5 \
    --batch_size 32 \
    --model_type bert \
    --learning_rate 0.00005 \
    --bert_learning_rate 0.00002 \
    --warmup_ratio 0.1 \
    --val_ratio 0.1 \
    --bert_seq_length 256 \
    --fc_size 768 \
    --dropout 0.1 \
    --grad_clip 1.0 \
    > logs/train_${DATE}_${TAG}.log 2>&1 &
