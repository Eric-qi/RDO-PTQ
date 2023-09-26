import io
import os
import copy
import yaml
import math
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from pytorch_msssim import ms_ssim


from pytorch_msssim import ms_ssim

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    msssim = ms_ssim(a, b, data_range=1.).item()
    return -10 * math.log10(1-msssim)

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def compute_loss(x,out_net,lamda):
    mse = torch.mean((x - out_net['x_hat'])**2).item()
    
    rate = compute_bpp(out_net)
    
    loss = rate + lamda * 255 *255 * mse
    return loss



def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def Test_kodak(model=None):
    
    testset_path = './datasets/kodak24'
    device = next(model.parameters()).device

    psnr_sum = 0.0
    msssim_sum = 0.0
    bit_sum = 0.0
    for i in range(len(os.listdir(testset_path))):
        #model = nic_tic(5, "mse", pretrained=True).eval()
        #model = copy.deepcopy(net)
        
        img = Image.open(testset_path+'/kodim'+str(i+1).zfill(2)+'.png').convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
        h, w = x.size(2), x.size(3)
        x_pad = pad(x, p)

        with torch.no_grad():
            out = model.forward(x_pad)

        rec = crop(out["x_hat"], (h,w))
        rec.clamp_(0, 1)      
        
#         print(i)
#         print(f'PSNR: {compute_psnr(x, rec):.4f}dB')
#         print(f'MS-SSIM: {compute_msssim(x, rec):.4f}dB')
#         print(f'Bit-rate: {compute_bpp(out):.4f} bpp')
        
        psnr_sum += compute_psnr(x, rec)
        msssim_sum += compute_msssim(x, rec)
        bit_sum += compute_bpp(out)
    
#     print("==================== KODAK24 ======================")
#     print(f'AVG PSNR: {psnr_sum/len(os.listdir(testset_path)):.2f}dB')
#     print(f'AVG MS-SSIM: {msssim_sum/len(os.listdir(testset_path)):.2f}dB')
#     print(f'AVG Bit-rate: {bit_sum/len(os.listdir(testset_path)):.4f} bpp') 
    
    logging.info("Test Data: Kodak24 with 512x768 ")
    logging.info(f'AVG PSNR: {psnr_sum/len(os.listdir(testset_path)):.2f}dB')
    logging.info(f'AVG MS-SSIM: {msssim_sum/len(os.listdir(testset_path)):.2f}dB')
    logging.info(f'AVG Bit-rate: {bit_sum/len(os.listdir(testset_path)):.4f} bpp')  