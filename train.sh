#!/bin/sh
python train.py --log_name "experiment_reppointsflow" \
--seg_lr 2e-5 --prior_lr 1e-7 --num_blocks 2 --batch_size 16 --epochs 100 --save_freq 3 \
--viz_freq 1 --log_freq 1 --gpu 0 --dims 64 --input_dim 2 --data_dir "/home/syk/cocostuff/dataset" --root_dir "./" \
--input_channels 3 --num_classes 80 --seed 1234 # to reproduce the result
