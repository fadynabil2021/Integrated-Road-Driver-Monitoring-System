#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

torchrun --master_port=9909 --master_addr=localhost --nproc_per_node=1 tools/train.py -c configs/rtdetrv2/config.yml --use-amp --seed=0 -r output/rtdetrv2_r18vd_120e_coco/best.pth --test-only 
