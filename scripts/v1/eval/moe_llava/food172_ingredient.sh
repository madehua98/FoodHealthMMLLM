# #!/bin/bash
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export WANDB_DISABLED=true
# export NCCL_IB_TIMEOUT=22
# export PYTHONPATH=/home/data_llm/madehua/FoodHealthMMLLM
# cd /home/data_llm/madehua/FoodHealthMMLLM/moellava/eval

# CONV="phi"
# CKPT="/media/fast_data/model/checkpoints/checkpoints-phi-2.7b-moe-v2k_0426/checkpoint-14500"
# model_name="MoE-LLaVA-Phi2-2.7B-4e-v2k_0426-checkpoint-14500"


# GPUs=("2" "3" "4" "6" "7") # Corrected array syntax
# question_path="/home/data_llm/madehua/FoodHealthMMLLM/eval/food2k/2k_questions.jsonl"
# answer_path="/home/data_llm/madehua/FoodHealthMMLLM/eval/food2k/answers"
# out_path="/home/data_llm/madehua/FoodHealthMMLLM/scripts/v1/eval/moe_llava"

# split_files=()
# while IFS= read -r line; do
#   split_files+=("$line")
# done < <(python split_jsonl.py --input_file $question_path --n ${#GPUs[@]})

# # Print the split files for debugging
# echo "Split files: ${split_files[@]}"


# cd /home/data_llm/madehua/FoodHealthMMLLM/scripts/v1/eval/moe_llava/
# # Loop through each GPU and execute the deepspeed.sh script with the respective split file
# for i in "${!GPUs[@]}"; do
#   gpu=${GPUs[$i]}
#   master_port=$((29500 + $gpu))  # or any other logic to generate a unique master port per GPU
#   question_file=${split_files[$i]}  # Get the corresponding split file for the current GPU
#   log_file="${out_path}/log_gpu${gpu}.txt"
#   nohup sh deepspeed.sh $CKPT $model_name $gpu $answer_path $master_port $question_file > $log_file 2>&1 &
# done

#!/bin/bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export NCCL_IB_TIMEOUT=22
export PYTHONPATH=/home/data_llm/madehua/FoodHealthMMLLM
cd /home/data_llm/madehua/FoodHealthMMLLM/moellava/eval

CONV="phi"
CHECKPOINT_DIR="/mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-moe-v172_ingredient_0426"
BASE_MODEL_NAME="MoE-LLaVA-Phi2-2.7B-4e-v172_ingredient_0426"

GPUs=("2" "3" "4" "6" "7")
question_path="/home/data_llm/madehua/FoodHealthMMLLM/eval/food172_ingredient/172_ingredient_questions.jsonl"
out_path="/home/data_llm/madehua/FoodHealthMMLLM/scripts/v1/eval/moe_llava"

# List of specific checkpoints to test
checkpoints_to_test=("checkpoint-344" "checkpoint-688" "checkpoint-1032" "checkpoint-1376")

split_files=()
while IFS= read -r line; do
  split_files+=("$line")
done < <(python split_jsonl.py --input_file $question_path --n ${#GPUs[@]})
# Print the split files for debugging
echo "Split files: ${split_files[@]}"

# Get a list of all checkpoint subdirectories
checkpoints=($(ls -d $CHECKPOINT_DIR/checkpoint-*))
base_port=29500  # Base port to start with for each checkpoint

cd /home/data_llm/madehua/FoodHealthMMLLM/scripts/v1/eval/moe_llava/
for CKPT in "${checkpoints[@]}"; do
  checkpoint_name=$(basename $CKPT)
  model_name="${BASE_MODEL_NAME}-${checkpoint_name}"
  
  # Construct the answer path for the current checkpoint
  answer_path="/home/data_llm/madehua/FoodHealthMMLLM/eval/food172_ingredient/answers"

  # Check if the current checkpoint is in the list of checkpoints to test
  if [[ " ${checkpoints_to_test[@]} " =~ " ${checkpoint_name} " ]]; then
    # Loop through each GPU and execute the deepspeed.sh script with the respective split file
    for i in "${!GPUs[@]}"; do
      gpu=${GPUs[$i]}
      master_port=$((base_port + $i))  # or any other logic to generate a unique master port per GPU
      question_file=${split_files[$i]}  # Get the corresponding split file for the current GPU
      log_file="${out_path}/logs/log_gpu${gpu}_${checkpoint_name}.txt"
      
      # Debugging prints
      echo "Running deepspeed.sh with the following parameters:"
      echo "Checkpoint: $CKPT"
      echo "Model Name: $model_name"
      echo "GPU: $gpu"
      echo "Answer Path: $answer_path"
      echo "Master Port: $master_port"
      echo "Question File: $question_file"
      echo "Log File: $log_file"

      nohup sh deepspeed.sh $CKPT $model_name $gpu $answer_path $master_port $question_file > $log_file 2>&1 &
    done
    wait
    base_port=$((base_port + ${#GPUs[@]}))
  fi
done



