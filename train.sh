#!/bin/sh
python train.py --log_name "experiment_reppointsflow" \
--lr 2e-5 --num_blocks 1 --batch_size 16 --epochs 150 --save_freq 10 \
--viz_freq 1 --log_freq 1 --gpu 0 --dims 128-128-128 --input_dim 2 --data_dir "/home/syk/cocostuff/dataset" --root_dir "./"
