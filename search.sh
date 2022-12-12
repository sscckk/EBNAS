#!/bin/bash

method="imagenet"
gpu="5,6,7"

# nohup python -W ignore train_search.py --save ${method} --gpu ${gpu} &

nohup python -W ignore train_search_imagenet.py --save ${method} --gpu ${gpu} &