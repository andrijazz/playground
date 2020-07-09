#!/bin/bash

# notes: ...
python train.py \
    --data_drive="/mnt" \
    --resume="" \
    --model="fcn8" \
    --dataset="kitti_semantics" \
    --batch_size=5 \
    --num_epochs=50 \
    --learning_rate=1e-3 \
    --gpu=1 \
    --keep_prob=0.5 \
    --weights="vgg16_weights.npz" \
    --save_step=500 \
    --val_step=5 \
    --debug_steps=1
