#!/bin/bash

# shellcheck disable=SC2164
export PYTHONPATH=/home/data_llm/madehua/FoodHealthMMLLM
moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01
CACHE_FOLDER="/media/fast_data/huggingface/hub/"
food_image_folder="/media/LLM_data/food_recognition_dataset"
json_folder="/mnt/data_llm/json_file"
version="172_ingredient_0426"
export NCCL_P2P_DISABLE=1
export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export NCCL_IB_TIMEOUT=22
gpus="1,2,3,4,6,7,8"
echo $gpus
deepspeed --include localhost:$gpus --master_port=2227 /home/data_llm/madehua/FoodHealthMMLLM/moellava/train/train_xformers.py \
    --do_train \
    --moe_enable True \
    --num_experts ${num_experts} \
    --top_k_experts ${top_k_experts} \
    --capacity_factor 1.5 \
    --moe_mode ${moe_mode} \
    --use_residual ${use_residual} \
    --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules fc1 fc2 wg \
    --deepspeed ../../zero2_offload.json \
    --model_name_or_path /mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-v0426 \
    --version phi \
    --data_path ${json_folder}/172_ingredient_train_prompt10.json \
    --image_folder ${food_image_folder} \
    --image_tower /mnt/data_llm/model/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --check_point_file_name /mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-moe-v${version}.json \
    --output_dir /mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-moe-v${version} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 64 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ${CACHE_FOLDER}
