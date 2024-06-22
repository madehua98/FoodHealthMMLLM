#!/bin/bash

# shellcheck disable=SC2164
export PYTHONPATH=/home/data_llm/FoodHealthMMLLM

CACHE_FOLDER="/media/fast_data/huggingface/hub/"
food_image_folder="/media/LLM_data/food_recognition_dataset"
json_folder="/mnt/data_llm/json_file"
check_point_name="v0608_6"
export NCCL_P2P_DISABLE=1
export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890
#export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export NCCL_IB_TIMEOUT=22
export MAX_JOBS=64

# --data_path ${json_folder}/101_train_prompt1.json \
#             ${json_folder}/train_ingredient_QA.json ${json_folder}/train_recipe_QA.json ${json_folder}/train_title_QA.json \
#             ${json_folder}/2k_train_prompt1.json \
#             ${json_folder}/172_train_prompt1.json ${json_folder}/172_ingredient_train_prompt1.json \
#             ${json_folder}/nutrition5k_train.json ${json_folder}/mix_food.json\
#             ${json_folder}/weight_dataset_train2.json ${json_folder}/train_nutrition_QA.json \
gpus="1,2,6,7,8,9"
deepspeed --include localhost:$gpus --master_port=22224 /home/data_llm/FoodHealthMMLLM/moellava/train/train_xformers.py \
    --deepspeed ../../zero2.json \
    --model_name_or_path microsoft/phi-2 \
    --version phi \
    --data_path ${json_folder}/stage2_240608.json \
    --image_folder ${food_image_folder} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /media/fast_data/huggingface/hub/models--LanguageBind--MoE-LLaVA-Phi2-Pretrain/snapshots/87dd7b7b768fbfbef94cec9dfd0bd04d2af4ca9d/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --check_point_file_name /mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-${check_point_name}.json \
    --output_dir /mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-${check_point_name} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 50 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 64 \
    --lazy_preprocess True \
    --cache_dir ${CACHE_FOLDER}

