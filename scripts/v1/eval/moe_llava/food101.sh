#!/bin/bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export NCCL_IB_TIMEOUT=22
export PYTHONPATH=/home/data_llm/madehua/FoodHealthMMLLM
cd /home/data_llm/madehua/FoodHealthMMLLM/moellava/eval

CONV="phi"
CKPT="/mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-moe-v101_0426/checkpoint-1419"
model_name="MoE-LLaVA-Phi2-2.7B-4e-v101_0426-checkpoint-1419"
prompt_num=10
GPUs=("1" "2" "3" "4")
question_path="/mnt/data_llm/json_file/101_questions.jsonl"
answer_path="/home/data_llm/madehua/FoodHealthMMLLM/eval/food101/answers"
out_path="/home/data_llm/madehua/FoodHealthMMLLM/scripts/v1/eval/moe_llava"


split_files=()
while IFS= read -r line; do
  split_files+=("$line")
done < <(python split_jsonl.py --input_file $question_path --n ${#GPUs[@]})

# Print the split files for debugging
echo "Split files: ${split_files[@]}"


cd /home/data_llm/madehua/FoodHealthMMLLM/scripts/v1/eval/moe_llava/
# Loop through each GPU and execute the deepspeed.sh script with the respective split file
for i in "${!GPUs[@]}"; do
  gpu=${GPUs[$i]}
  master_port=$((29500 + $gpu))  # or any other logic to generate a unique master port per GPU
  question_file=${split_files[$i]}  # Get the corresponding split file for the current GPU
  log_file="${out_path}/log_gpu${gpu}.txt"
  nohup sh deepspeed.sh $CKPT $model_name $gpu $answer_path $master_port $question_file > $log_file 2>&1 &
done

