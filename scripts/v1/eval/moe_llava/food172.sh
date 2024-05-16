#!/bin/bash


export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export NCCL_IB_TIMEOUT=22
export PYTHONPATH=/home/data_llm/madehua/FoodHealthMMLLM
CONV="phi"
CKPT="/mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-moe-v172_0426"
model_name="MoE-LLaVA-Phi2-2.7B-4e-v172_0426"
question_path="/mnt/data_llm/json_file/172_questions.jsonl"
answer_path="/home/data_llm/madehua/FoodHealthMMLLM/eval/food172/answers"
deepspeed --include localhost:0 --master_port=2226 moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file $question_path \
    --image-folder '' \
    --answers-file $answer_path/${model_name}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

# python3 scripts/convert_vizwiz_for_submission.py \
#     --annotation-file ${EVAL}/vizwiz/llava_test.jsonl \
#     --result-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
#     --result-upload-file ${EVAL}/vizwiz/answers_upload/${CKPT_NAME}.json