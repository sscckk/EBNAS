## Introduction
**EBNAS** is a differentiable architecture search method for binary networks. Corresponding improvements are proposed to deal with the information loss due to binarization, the discrete error between search and evaluation, and the unbalanced operation advantage in the search space. With a similar number of model parameters, EBNAS outperforms other binary NAS methods in terms of accuracy and efficiency. Compared with manually designed binary networks, it remains competitive.

This code is based on the implementation of  [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).

## Results
### Results on CIFAR10
Method | Params(M) | Error(%)| Search-Cost
--- | --- | --- | ---
ResNet18 | 11.2 | 7.0 | -
BNAS-C  | 42.4 | 5.57 | 0.4
CP-NAS | 10.6 | 4.73 | 0.1
BATS  | 10.0 | 4.5 | 0.25
EBNAS    | 10.0 | **4.39** |0.04

### Results on CIFAR100
Method | Params(M) | Error(%)| Search-Cost
--- | --- | --- | ---
ResNet18 | 11.2 | 24.39 | -
BATS  | 10.0 | 24.3 | 0.25
EBNAS    | 10.0 | **21.90** |0.04

### Results on ImageNet
Method | OPs(1e8) |Top-1 Error(%)|Top-5 Error(%)| Search-Cost
--- | --- | --- | --- | ---
ResNet18 | 18.2 | 30.4 | 10.8 | -
BNAS-H  | 4.71 | 36.49 | 16.09 | 0.4
CP-NAS** | 2.58 | 33.5 | 13.2 | 0.1
BATS(2x) | 1.55 | 33.9 | 13.0 | 0.25
EBNAS    | 1.72 | **32.2** | **12.6** | 0.04

## Environment
Python3 with Pytorch(1.10.1)

GPU: NVIDIA RTX 3090

## Usage
#### Search on CIFAR10

```
python train_search.py \\
```

#### Evaluation

The evaluation process contains two stages of full-precision activation/binary weight and binary activation/binary weight.

Use --binary to select which stage to run.

##### Here is the evaluation on CIFAR10/100:

```
python train.py \\
       --auxiliary \\
       --cutout \\
       --binary \\
```

##### Here is the evaluation on ImageNet (mobile setting):
```
python train_imagenet.py \\
       --tmp_data_dir /path/to/your/data \\
       --save log_path \\
       --auxiliary \\
       --binary \\
       --note note_of_this_run
```
