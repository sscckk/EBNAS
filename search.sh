#!/bin/bash

method="EBNAS"
gpu="5"

nohup python -W ignore train_search.py --save ${method} --gpu ${gpu} &

