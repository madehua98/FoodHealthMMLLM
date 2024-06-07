#!/bin/bash
export PYTHONPATH=/home/data_llm/FoodHealthMMLLM

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01
JSON_FOLDER="ft_json"
IMAGE_FOLDER="train_image_video"
check_point_name="nutv2"
CACHE_FOLDER="/media/fast_data/huggingface/hub/"
json_folder="/mnt/data_llm/json_file"

#cd ~/MoE-LLaVA
export PYTHONPATH=/home/xuzhenbo/FoodHealthMMLLM
export NCCL_P2P_DISABLE=1
export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890
#export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export NCCL_IB_TIMEOUT=22
export MAX_JOBS=32

gpus="4,6,8,9"
deepspeed --include localhost:$gpus --master_port=2224 /home/data_llm/FoodHealthMMLLM/moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules fc1 fc2 wg \
    --deepspeed ../../zero2_offload.json \
    --model_name_or_path /mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-${check_point_name} \
    --version phi \
    --data_path ${json_folder}/weight_dataset_train2.json ${json_folder}/nutrition5k_train.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --check_point_file_name /mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-${check_point_name}.json \
    --output_dir /mnt/data_llm/model/checkpoints/llavaphi-2.7b-finetune-moe-${check_point_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1200 \
    --gradient_checkpointing True \
    --dataloader_num_workers ${MAX_JOBS} \
    --lazy_preprocess True \
    --report_to none \
    --cache_dir ${CACHE_FOLDER}
