#!/bin/bash

# 分布式训练参数
NUM_GPUS=4  # 使用的GPU数量
BATCH_SIZE=196  # 全局 batch size
DATA_PATH="/fs/scratch/PAS2473/ICML2025/dataset/openimage"  # 数据集路径
VAE_PATH="/fs/scratch/PAS2473/ICML2025/hart/hart/hart-0.7b-1024px/tokenizer"  # 预训练VAE路径
RESULTS_DIR="/fs/scratch/PAS2473/ICML2025/result/vqvae/256"  # 结果保存路径
CLOUD_SAVE_PATH="/fs/scratch/PAS2473/ICML2025/logs/vqgan/256"  # 云端保存路径

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定使用的4张卡

# 启动训练
torchrun --nproc_per_node=$NUM_GPUS --master_port=12346 \
    /fs/scratch/PAS2473/ICML2025/dart/training/vq_train.py \
    --data-path "$DATA_PATH" \
    --cloud-save-path "$CLOUD_SAVE_PATH" \
    --results-dir "$RESULTS_DIR" \
    --vq-model "DART_tokenizer" \
    --vae-path "$VAE_PATH" \
    --epochs 2 \
    --lr 1e-4 \
    --global-batch-size $BATCH_SIZE \
    --global-seed 42 \
    --mixed-precision "none" \
    --image-size 256 \
    --disc-start 2000 \
    --disc-weight 0.5 \
    --disc-type "patchgan" \
    --disc-loss "hinge" \
    --gen-loss "hinge" \
    --perceptual-weight 1.0 \
    --reconstruction-weight 1.0 \
    --reconstruction-loss "l2" \
    --codebook-weight 1.0 \
    --num-workers 10 \
    --finetune-decoder \