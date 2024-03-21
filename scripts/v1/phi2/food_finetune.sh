#!/bin/bash

CACHE_FOLDER="/media/fast_data/huggingface/hub/"
food_image_folder="/media/LLM_data/food_recognition_dataset"
json_folder="/mnt/data_llm/json_file"
check_point_name="v1"
export NCCL_P2P_DISABLE=1
export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890
cd ~/MoE-LLaVA && export PYTHONPATH=/home/xuzhenbo/MoE-LLaVA
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
cd /home/data_llm/FoodHealthMMLLM
deepspeed --include localhost:0,1,2,3,4,8,9 --master_port=2222 moellava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path microsoft/phi-2 \
    --version phi \
    --data_path ${json_folder}/101_train_prompt1.json \
                ${json_folder}/train_ingredient_QA.json ${json_folder}/train_recipe_QA.json ${json_folder}/train_title_QA.json \
                ${json_folder}/2k_train_prompt1.json \
                ${json_folder}/172_train_prompt1.json ${json_folder}/172_ingredient_train_prompt1.json \
                ${json_folder}/nutrition5k_train.json ${json_folder}/mix_food.json\
    --image_folder ${food_image_folder} \
    --image_tower /media/LLM_data/model/openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /media/fast_data/huggingface/hub/models--LanguageBind--MoE-LLaVA-Phi2-Pretrain/snapshots/87dd7b7b768fbfbef94cec9dfd0bd04d2af4ca9d/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --check_point_file_name /media/LLM_data/model/moellava/checkpoints/checkpoints-phi-2.7b-${check_point_name}.json \
    --output_dir /media/LLM_data/model/moellava/checkpoints/checkpoints-phi-2.7b-${check_point_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --cache_dir ${CACHE_FOLDER}

