#!/bin/bash

CACHE_FOLDER="/home/xuzhenbo/.cache/huggingface/hub/"
IMAGE_FOLDER="/home/xuzhenbo/.cache/huggingface/hub/datasets--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b"
JSON_FOLDER="/home/xuzhenbo/.cache/huggingface/hub/datasets--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json"
export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890
cd ~/MoE-LLaVA && export PYTHONPATH=/home/xuzhenbo/MoE-LLaVA
export NCCL_P2P_DISABLE=1

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --num_gpus=2 moellava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen-1_8B \
    --version qwen \
    --data_path ${JSON_FOLDER}/la_tune_256k.json \
                ${JSON_FOLDER}/lrv_tune_331k.json ${JSON_FOLDER}/lvis_tune_220k_.json \
                ${JSON_FOLDER}/svit_tune_157k.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/llavaqwen-1.8b-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavaqwen-1.8b-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ${CACHE_FOLDER}

