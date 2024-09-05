#!/bin/bash
PRETRAIN_NAME=MGM-2B-Pretrain
FINETUNE_NAME=Parsa
AUX_SIZE=768

deepspeed mgm/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /content/MGM/work_dirs/MGM/MGM-2B \
    --version gemma \
    --data_path /content/MGM/dataset.json \
    --image_folder /content/MGM/content/content/combined/CXR_png \
    --vision_tower model_zoo/OpenAI/clip-vit-large-patch14-336 \
    --vision_tower_aux model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --image_size_aux $AUX_SIZE \
    --bf16 True \
    --output_dir ./work_dirs/$FINETUNE_NAME \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
