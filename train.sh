#!/bin/sh
python train.py --log_name "experiment_reppointsflow" \
--lr 1e-4 --num_blocks 2 --batch_size 64 --epochs 100 --save_freq 5 \
--viz_freq 1 --log_freq 1 --gpu 0 --dims 32 --input_dim 2 --data_dir "/media/mlsyn91/43c99d24-729f-4905-9c48-c16b6d7df148/coco/cocostuff/dataset" --root_dir "./" \
--input_channels 3 --num_classes 80 --seed 1234 # to reproduce the result
