# Task-Oriented PTQ
We release a more lightweight task-oriented posting-training quantization (PTQ) for learned image compression (LIC), based on the original paper. This version adopts a more general quantization architecture for further study and can greatly reduce optimization time.


## TODO
✅ release related codes

<input type="checkbox" disabled /> support multi-bit layer-wise activation quantization to reduce coding time;

<input type="checkbox" disabled /> optimize activation quantization;

<input type="checkbox" disabled /> check the correction of codes;

<input type="checkbox" disabled /> add other models;

<input type="checkbox" disabled /> update other PTQ methods for LICs;

<input type="checkbox" disabled /> ...



## Code

* main.py：quantize and optimize FP32 models

* test_datasets.ipynb：evaluate image sets on various models

* BD-rate.ipynb：compute BD-rate

* config.yaml：model defination and optimization setting

* ckpts: contains pretrained FP32 model checkpoints

* results: contains all experimental results

## Environment
```bash
        pip install -r requirements.txt
```

## Results
* log: the log of quantization and evaluation

* outputs: output models saved dir

* config

## Usage

> **Note 1**: For MSE, lambda is chosen from **{0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483}**;

> **Note 2**: For MS-SSIM, lambda is chosen from **{2.40, 4.58, 8.73, 16.64, 31.73, 60.50}**;

###  Train Lu2022



1. MSE
```bash
CUDA_VISIBLE_DEVICES=0 python main2.py --arch Lu2022 --quality 6 --lmbda 0.0483 --save --n_bits_w 8 \
    --channel_wise --type mse --task_loss 2 --init max  --config ./config.yaml
```


2. MS-SSIM

```bash
CUDA_VISIBLE_DEVICES=1 python main2.py --arch Lu2022 --quality 6 --lmbda 60.50 --save  --n_bits_w 8 \
    --channel_wise --type ms-ssim --task_loss 2 --init max  --config ./config.yaml
```


###  Train Cheng2020

1. MSE
```bash
CUDA_VISIBLE_DEVICES=2 python main2.py --arch Cheng2020 --quality 6 --lmbda 0.0483 --save --n_bits_w 8 \
    --channel_wise --type mse --task_loss 2 --init max  --config ./config.yaml
```


2. MS-SSIM
```bash
CUDA_VISIBLE_DEVICES=3 python main2.py --arch Cheng2020 --quality 6 --lmbda 60.50 --save --n_bits_w 8 \
    --channel_wise --type ms-ssim --task_loss 2 --init max  --config ./config.yaml
```
