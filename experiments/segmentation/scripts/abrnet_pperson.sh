# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pperson \
    --model abrnet --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname abrnet_res101_pperson --dilated --epochs 150 --batch-size 20 --lr 0.01

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pperson \
    --model abrnet --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pperson/abrnet/abrnet_res101_pperson/model_best.pth.tar --split val --mode testval --dilated

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pperson \
    --model abrnet --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pperson/abrnet/abrnet_res101_pperson/model_best.pth.tar --split val --mode testval --ms --dilated