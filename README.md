# Rate-Distortion Optimized Post-Training Quantization for Learned Image Compression




## Overview
This repository contains the simple implementation of PTQ for LIC.

Our goal is to provide some materials and data that are easy to use for further study. Some errors are inevitable.



## 1.&ensp;Literature Comparison
### We researched most of works on quantization of LIC as we can. These works containg:
* [[ICLR 2019]](https://openreview.net/forum?id=S1zz2i0cY7) `Ballé et al., 2019` : Interger Networks for Data Compression with Latent-Variable Models
* [[ICIP 2020]](https://ieeexplore.ieee.org/abstract/document/9190805) `Sun et al., 2020` : End-to-End Learned Image Compression with Fixed Point Weight Quantization
* [[TCSVT 2020]](https://ieeexplore.ieee.org/abstract/document/9270012) `Hong et al., 2020` : Efficient Neural Image Decoding via Fixed-Point Inference
* [[PCS 2021]](https://ieeexplore.ieee.org/abstract/document/9477496)  `Sun et al., 2021` : Learned Image Compression with Fixed-point Arithmetic
* [[Arxiv 2021]](https://arxiv.org/abs/2111.09348) `Sun et al., 2021*` : End-to-End Learned Image Compression with Quantized Weights and Activations
* [[Arxiv 2022]](https://arxiv.org/abs/2202.07513) `He et al., 2022` : Post-Training Quantization for Cross-Platform Learned Image Compression
* [[PCS 2022]](https://ieeexplore.ieee.org/abstract/document/10018040) `Koyuncu et al., 2022` : Device Interoperability for Learned Image Compression with Weights and Activations Quantization
* [[TCSVT2022]](https://ieeexplore.ieee.org/abstract/document/9997555) `Sun et al., 2022` : Q-LIC: Quantizing Learned Image Compression with Channel Splitting
* [[Arxiv 2023]](https://arxiv.org/pdf/2211.02854v2.pdf) `Shi et al., 2023` : Rate-Distortion Optimized Post-Training Quantization for Learned Image Compression
* [`updating`]()

### Results of quantizing LIC in terms of BD-rate.

|        Methods       | Bit-Width (W/A) | Granularity  |  Type  |           Models          |  Kodak24 | 
| :------------------: | :-------------: | :----------: | :----: | :-----------------------: | :------: | 
| `Ballé et al., 2019` |       None      |     None     |   QAT  |        Ballé2018          |   None   |
|  `Sun et al., 2020`  |       8/32      | channel-wise |   QAT  |        Cheng2019          |   None   |
|  `Hong et al., 2020` |       8/10      |  layer-wise  |   QAT  |  Ballé2018 <br> Chen2021  |  26.50% <br> 16.04%   |
|  `Hong et al., 2020` |       8/16      |  layer-wise  |   QAT  |  Ballé2018 <br> Chen2021  |  17.90% <br> 3.25%    |
|  `Sun et al., 2021`  |       8/32      | channel-wise |   QAT  |        Cheng2019          |          None         |
|  `Sun et al., 2021*` |       8/8       | channel-wise |   QAT  |        Cheng2019          |          None         |
|   `He et al., 2022`  |       8/8       | layer-wise (W) <br> channel-wise(A) |   PTQ  | Balle2018 <br> Minnen2018 <br> Cheng2020 |  None <br> 0.66% <br> 0.42% |
|`Koyuncu et al., 2022`|      16/16      | channel-wise |   PTQ  |         TEAM14            |          0.29%        |
|  `Sun et al., 2022`  |       8/8       | channel-wise |   PTQ  | Cheng2019 <br> Cheng2020   |  4.98% & 4.34% (MS-SSIM) <br> 10.50% & 4.40% (MS-SSIM)        |
|  `Shi et al., 2022`  |       8/8       | channel-wise |   PTQ  | Minnen2018 <br> Cheng2020 <br> Lu2022  |  5.84% <br> 4.88% <br> 3.70%   |
|  `Shi et al., 2022`  |      10/10      | channel-wise |   PTQ  | Minnen2018 <br> Cheng2020 <br> Lu2022  |  0.41% <br> 0.43% <br> 0.49%| 1.22% <br> 0.65% <br> 1.03% |

### The notion of above table mentioned:
* PTQ : Posting Training Quantization
* QAT : Quantization Aware Training


### The models of above table mentioned:
* [[ICLR 2018]](https://arxiv.org/abs/1802.01436) `Ballé2018` : Variational Image Compression with a Scale Hyperprior
* [[NeurIPS 2018]](https://proceedings.neurips.cc/paper/2018/hash/53edebc543333dfbf7c5933af792c9c4-Abstract.html) `Minnen2018` : Joint Autoregressive and Hierarchical Priors for Learned Image Compression
* [[CVPR Workshop 2019]](https://arxiv.org/abs/1906.09731) `Cheng2019` : Deep Residual Learning for Image Compression
* [[CVPR 2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_Learned_Image_Compression_With_Discretized_Gaussian_Mixture_Likelihoods_and_Attention_CVPR_2020_paper.html) `Cheng2020` : Learned Image Compression With Discretized Gaussian Mixture Likelihoods and Attention Modules
* [[TIP 2021]](https://ieeexplore.ieee.org/document/9359473) `Chen2021`: End-to-End Learnt Image Compression via Non-Local Attention Optimization and Improved Context Modeling
* [[JPEG AI CfP 2022]](https://jpeg.org/jpegai/) `TEAM14` : Presentation of the Huawei response to the JPEG AI Call for Proposals: Device agnostic learnable image 								coding using primary component extraction and conditional coding
* [[Arxiv 2022]](https://arxiv.org/abs/2204.11448) `Lu2022` : High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation
* [[DCC 2022]](https://arxiv.org/abs/2111.06707) `Lu2022*` : Transformer-based Image Compression

### The Datasets of above table mentioned:
* [`Kodak`](https://r0k.us/graphics/kodak/) : with 24 image resolution at 768×512 
* [`Tecnick`](https://tecnick.com/?aiocp) : with 100 image resolution at 1200 × 1200
* [`CLIC`](http://compression.cc/tasks/#image) :  CLIC professional validation dataset contains 41 images at 2k spatial resolution approximately

### Note:
1) All of data are obtained from their avaliable published paper
2) Only the primary data of the paper is displayed here, and the reproduced data by others is not included
3) The *None* in table means that authors did these related experiments without showing the BD-rate




## 2.&ensp;PTQ optimization

PTQ has attracted a lot of attention. More and more wroks try to push the limits of PTQ. Here we introduce some novel works of PTQ.

### Task-oriented optimization
Recently, a lot of works have recognized that minimizing quantization error may not be optimal. We should pay more attention to the objects of tasks, e.g., the accuracy, the PSNR, the MS-SSIM. Therefore, these works push the limit of PTQ by minimizing task loss. We call this idea **Task-oriented optimization**.  Here are some works about this idea. These works containg:

* [[PMLR 2020]](https://proceedings.mlr.press/v119/nagel20a) `AdaRound` : Up or Down? Adaptive Rounding for Post-Training Quantization
* [[Arxiv 2020]](https://arxiv.org/abs/2006.10518) `AdaQuant` : Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming
* [[ML 2021]](https://link.springer.com/article/10.1007/s10994-021-06053-z) `LAPQ` : Loss aware post-training quantization`
* [[ICLR 2021]](https://arxiv.org/abs/2102.05426) `BRECQ` : BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction
* [[ICLR 2022]](https://arxiv.org/abs/2203.05740) `QDrop` : QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization
* [[CVPR 2023]](https://arxiv.org/abs/2212.07048) `PD-Quant` : PD-Quant: Post-Training Quantization Based on Prediction Difference Metric
* [[Arxiv 2023]](https://arxiv.org/abs/2208.11945) `AQuant` : Efficient Adaptive Activation Rounding for Post-Training Quantization
* [[updating]]()




## 3.&ensp;Usage

* Environment
```bash
        pip install -r requirements.txt
```

* light uniform PTQ

* task-oriented PTQ

* ...



## Citation

If you use this project, please considering citing the relevant original publications for the models and datasets, and citing the paper as:

	@ARTICLE{10274709,
  		author={Shi, Junqi and Lu, Ming and Ma, Zhan},
  		journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  		title={Rate-Distortion Optimized Post-Training Quantization for Learned Image Compression}, 
  		year={2023},
  		volume={},
  		number={},
  		pages={1-1},
  		doi={10.1109/TCSVT.2023.3323015}
    	}





## Acknowledgement
This framework is based on [BRECQ](https://github.com/yhhhli/BRECQ), [CompressAI](https://github.com/InterDigitalInc/CompressAI/), and [TinyLIC](https://github.com/lumingzzz/TinyLIC).

We thank the authors for sharing their codes.
