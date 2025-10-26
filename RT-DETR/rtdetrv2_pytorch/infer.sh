#!/bin/bash

python references/deploy/rtdetrv2_torch.py --config configs/rtdetrv2/config.yml --im-file "$1" --resume output/rtdetrv2_r18vd_120e_coco/best.pth --img-out "$2" --device cuda

python view.py "$2"
