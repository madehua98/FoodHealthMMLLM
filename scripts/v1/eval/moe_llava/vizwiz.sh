#!/bin/bash

CONV="phi"
CKPT="/mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-v1-moe-v1"
question_path="/mnt/data_llm/json_file/101_questions.jsonl"
answer_path = "/home/data_llm/madehua/FoodHealthMMLLM/eval/food101/answers/"
deepspeed --include localhost:1 --master_port=2224 moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file $question_path \
    --image-folder '' \
    --answers-file .answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

# python3 scripts/convert_vizwiz_for_submission.py \
#     --annotation-file ${EVAL}/vizwiz/llava_test.jsonl \
#     --result-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
#     --result-upload-file ${EVAL}/vizwiz/answers_upload/${CKPT_NAME}.json