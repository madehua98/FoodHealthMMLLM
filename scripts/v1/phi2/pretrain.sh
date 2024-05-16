#!/bin/bash

CACHE_FOLDER="/home/xuzhenbo/.cache/huggingface/hub/"
IMAGE_FOLDER="/home/xuzhenbo/.cache/huggingface/hub/datasets--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b"
JSON_FOLDER="/home/xuzhenbo/.cache/huggingface/hub/datasets--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json"
export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_P2P_DISABLE=1
cd ~/MoE-LLaVA && export PYTHONPATH=/home/xuzhenbo/MoE-LLaVA

#HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
#nohup deepspeed --include localhost:0 moellava/train/train_mem.py \
nohup deepspeed --include localhost:1,2 moellava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path microsoft/phi-2 \
    --version plain \
    --data_path ${JSON_FOLDER}/llava_image_.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower google/siglip-so400m-patch14-384 \
    --image_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llavaphi-2.7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none \
    --cache_dir ${CACHE_FOLDER} > phi2_pretrain.out &



