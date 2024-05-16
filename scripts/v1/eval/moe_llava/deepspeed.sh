#!/bin/bash


export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export NCCL_IB_TIMEOUT=22
export PYTHONPATH=/home/data_llm/madehua/FoodHealthMMLLM
CONV="phi"
CKPT=$1
model_name=$2
gpu=$3
answer_path=$4
master_port=$5
question_path=$6

cd /home/data_llm/madehua/FoodHealthMMLLM/moellava/eval
deepspeed --include localhost:$gpu --master_port=$master_port test_food.py \
    --model-path ${CKPT} \
    --question-file $question_path \
    --image-folder '' \
    --answers-file $answer_path/${model_name}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}