# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_parse --dataset pperson \
    --model abrnet --aux --base-size 473 --crop-size 473 \
    --backbone resnet101 --checkname abrnet_res101_pperson --dilated --epochs 150 --batch-size 16 --lr 0.01

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pperson \
    --model abrnet --aux --base-size 473 --crop-size 473 \
    --backbone resnet101 --resume experiments/segmentation/runs/pperson/abrnet/abrnet_res101_pperson/model_best.pth.tar --split val --mode testval --dilated

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pperson \
    --model abrnet --aux --base-size 473 --crop-size 473 \
    --backbone resnet101 --resume experiments/segmentation/runs/pperson/abrnet/abrnet_res101_pperson/model_best.pth.tar --split val --mode testval --ms --dilated