#!/bin/bash

# notes: ...
python train.py \
    --batch_size 2 \
    --dataset cityscapes \
    --learning_rate 0.00001 \
    --num_epochs 50 \
    --keep_prob 0.5 \
    --init_weights True \
    --gpu 1
