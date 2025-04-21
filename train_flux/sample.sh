#!/usr/bin/env sh

CUDA_VISIBLE_DEVICES=2 python sample.py \
    --model_name flux \
    --step 30 \
    --condition_size 512 \
    --target_size 1024 \
    --task_name geneval \
    --lora_dir /mnt/petrelfs/zhuole/ReflectionFlow/train_flux/runs/full_data_v4_cond_512/20250410-191141/ckpt/16000/pytorch_lora_weights.safetensors \
    --output_dir /mnt/petrelfs/zhuole/ReflectionFlow/train_flux/samples/full_data_v4_512_16k \
    --seed 0 \
