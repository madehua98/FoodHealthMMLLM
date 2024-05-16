#!/bin/bash

CACHE_FOLDER="/media/fast_data/huggingface/hub/"
IMAGE_FOLDER="/media/fast_data/huggingface/hub/datasets--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b"
JSON_FOLDER="/media/fast_data/huggingface/hub/datasets--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json"
export NCCL_P2P_DISABLE=1
export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890
cd ~/MoE-LLaVA && export PYTHONPATH=/home/xuzhenbo/MoE-LLaVA
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --num_gpus=10 moellava/train/train_xformers.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path microsoft/phi-2 \
    --version phi \
    --data_path ${JSON_FOLDER}/la_tune_256k.json \
                ${JSON_FOLDER}/lrv_tune_331k.json ${JSON_FOLDER}/lvis_tune_220k_.json \
                ${JSON_FOLDER}/svit_tune_157k.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower /media/LLM_data/model/openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /media/fast_data/huggingface/hub/models--LanguageBind--MoE-LLaVA-Phi2-Pretrain/snapshots/87dd7b7b768fbfbef94cec9dfd0bd04d2af4ca9d/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /media/LLM_data/model/moellava/checkpoints/llavaphi-2.7b-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
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
    --tf32 False \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --cache_dir ${CACHE_FOLDER}

