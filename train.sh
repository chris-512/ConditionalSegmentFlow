#!/bin/sh
python train.py --log_name "experiment_distflow" \
--lr 2e-5 --num_blocks 1 --batch_size 16 --epochs 150 --save_freq 10 \
--viz_freq 1 --log_freq 1 --gpu 0 --dims 128-128-128 --input_dim 2 --root_dir "/media/mlsyn91/524fec07-b23a-4712-8453-221661364c71/data/distflow"
