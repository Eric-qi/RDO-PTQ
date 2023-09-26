# Light Uniform PTQ


## TODO
✅ update light uniform PTQ codes

<input type="checkbox" disabled /> update solutions for loading INT8 model from INT8.pt;

<input type="checkbox" disabled /> check the correction of codes;

<input type="checkbox" disabled /> update other light PTQ methods;

<input type="checkbox" disabled /> ...


## Function
* quantization: quantize FP32 to INT8 and FP16.
* evaluation

## Models
* FP32: 32-bit floating-point

* FP16：16-bit floating-point

* INT8：8-bit fixed-point
    1. weight:
            INT8 channel-wise PTQ
    2. activation:
            16-bit layer-wsie dynamic PTQ for acceleration

## Code

* quantize.py：quantize FP32 models to FP16 & INT8

* single_test.py: evaluate single image on various models

* dataset_test.py：evaluate image sets on various models

* pretrained: contains pretrained FP32 models

* quant_model.py: fully INT8 model

* quant_coding_model.py: only entropy coding quantized INT8 model

* results: contains all experimental results

## Environment
```bash
        pip install -r requirements.txt
```

## Results
* log: the log of quantization and evaluation

* outputs: output saved dir, including images and models

* FP32 & FP16 & INT8: the dir of saved reconstruction images


## Usage

1. quantize.py: to quantize FP32 models to INT8 or FP16

```bash
        python quantize.py --type INT8
```
```bash
        python quantize.py --type FP16
```
2. single_test.py: to test various models (e.g., FP32, FP16, INT8) on a single image
```bash
        python single_test.py --type FP32 --gt_path './data/2K.png' --lrd 0.0008
```
* --type: FP32, FP16, INT8
* --gt_path: the path to gorund truth image
* --lrd: a hyper-parameter to control bit-rate, positive correlation, suggest [0.00005, 0.0009]

3. dataset_test.py: to test various models (e.g., FP32, FP16, INT8) on a dataset
```bash
        python dataset_test.py --type FP32 --gt_path './data/Kodak' --lrd 0.0008 --freq 10
```
* --type: FP32, FP16, INT8
* --gt_path: the path to gorund truth images folder
* --lrd: a parameter to control bit-rate, positive correlation, suggest [0.00005, 0.0009]
* --freq: the prequence of print and save; if you need to save all images, please set **--freq 1**;

> **Note 1**: We adopt fully quantized model for INT8. Besides, only quantizing entropy coding is also supported, where you can use quant_coding_model.py instead of quant_model.py. Generally, only quantizing entropy coding can get better performance.

> **Note 2**: Here, models are saved in INT8 or FP16, but we adopt fake quantization for forward propagation to avoid some operations that are not supported by INT or hardware. Thus, the coding times of the three models are similar.

> **Note 3**: For INT8 evaluation, the INT8 model is from the current quantization output instead of INT8.pt, which is due to the fact that loading INT8.pt directly will cause a pytorch error. This error can be resolved by revising the underlying pytorch code, and we will update related solutions.