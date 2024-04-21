#!/bin/bash

# Script to run main.py with specified hyperparameters

python main.py \
  --data_path "E:\download\data" \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --epochs 20 \
  --w_decay 0.00001 \
  --step_size 50 \
  --gamma 0.5 \
  --base_model "DeepLabv3Plus" \
  --loss "CrossEntropy"