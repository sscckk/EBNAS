#!/bin/bash

dataset="imagenet"
layers="14"
init_channels="96"
arch="s91_16b2"
suf="s91_16b2-14-96"
gpu="0,1"

# nohup python -W ignore train_cifar10.py --layers ${layers} --init_channels ${init_channels} --arch ${arch} --save cifar10-${arch}-${layers}-${init_channels} --cutout --auxiliary --gpu ${gpu} --binary &

# nohup python -W ignore train_cifar100.py --layers ${layers} --init_channels ${init_channels} --arch ${arch} --save cifar100-${arch}-${layers}-${init_channels} --mixup --auxiliary --gpu ${gpu} --binary &

nohup python -W ignore train_imagenet.py --layers ${layers} --init_channels ${init_channels} --arch ${arch} --save imagenet-${arch}-${layers}-${init_channels} --auxiliary --gpu ${gpu} --binary &
