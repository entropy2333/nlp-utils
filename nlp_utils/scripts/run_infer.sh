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
python -u inference.py \
    --bert_dir hfl/chinese-roberta-wwm-ext \
    --load_model_path 'save/20220608_204445/model_epoch_2_mean_f1_0.9999.bin' \
    --test_output_csv outputs/${DATE}_${TAG}.csv \
    --bert_seq_length 256 \
    --dropout 0.1 \
