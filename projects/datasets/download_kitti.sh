#!/usr/bin/env bash

# downloading ...
echo "downloading kitti..."
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip

# unpacking
echo "unpacking kitti..."
unzip data_semantics.zip -d data_semantics

# create val set
mkdir -p data_semantics/val/{image_2,semantic_rgb}
cp data_semantics/training/image_2/000160_10.png data_semantics/val/image_2/000160_10.png
cp data_semantics/training/semantic_rgb/000160_10.png data_semantics/val/semantic_rgb/000160_10.png
