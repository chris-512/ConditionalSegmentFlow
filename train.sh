#!/bin/sh
python train.py --log_name "experiment_reppointsflow" \
--lr 1e-4 --num_blocks 1 --batch_size 32 --epochs 100 --save_freq 5 \
--viz_freq 1 --log_freq 1 --gpu 0 --dims 128 --input_dim 2 --data_dir "/home/syk/cocostuff/dataset" --root_dir "./" \
--input_channels 3 --num_classes 80 --seed 1234 # to reproduce the result
