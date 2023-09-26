import os
import sys
import math
import time
import pickle
import random
import argparse
import warnings
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from pytorch_msssim import ms_ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from models.tinylic import TinyLIC
from quant_int import *
from utils import *

device = 'cuda'
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='running parameters',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# general parameters for data and model
parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str, help='result dir name')
parser.add_argument('--save', default=True, help='save quantized model')
parser.add_argument('--fp32_name',default='tinylic', help='fp32_model_path')

# quantization parameters
parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
parser.add_argument('--n_bits_a', default=16, type=int, help='bitwidth for activation quantization')
parser.add_argument('--act_quant', default=True, help='apply activation quantization')
parser.add_argument('--test_before_calibration', default=True, type=bool, help='test_before_calibration')
parser.add_argument('--sym', default=True, help='symmetric reconstruction')

args = parser.parse_args(args=[])


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False  # may slow
    torch.backends.cudnn.deterministic = True

def validate_model(model):
    model.eval()
    device = next(model.parameters()).device
    
    sum_psnr = 0.0
    sum_msssim = 0.0
    sum_bpp = 0.0
    
    lambda_rd = torch.tensor([0.0007]).to(device)
    img_num = 1
    for i in range(1):
        img = Image.open('./data/Kodak/kodim'+str(i+1).zfill(2)+'.png').convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        
        p = 64
        h, w = x.size(2), x.size(3)
        x_pad = pad(x, p)
        
        with torch.no_grad():

            out_enc = model.compress(x_pad, lambda_rd)
            out = model.decompress(out_enc["strings"], out_enc["shape"], lambda_rd)
            rec = crop(out['x_hat'], (h, w))

        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels


        sum_psnr += compute_psnr(x, rec)
        sum_msssim += compute_msssim(x, rec)
        sum_bpp += bpp

    return sum_psnr/img_num, sum_msssim/img_num , sum_bpp/img_num


def generator(qnn, args):
    device = next(qnn.parameters()).device
    img = Image.open('./data/Kodak/kodim'+str(23).zfill(2)+'.png').convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    p = 64
    h, w = x.size(2), x.size(3)
    x_pad = pad(x, p)
    lambda_rd = torch.tensor([0.0005]).to(device)
    
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, args.act_quant)
    _ = qnn(x_pad, lambda_rd)

    return qnn


def quantize_int8(args, fp32_name, output_dir):
    """
    Quantize a FP32 Models to INT8 Models;
    Based on Post-Training-Quantization (PTQ):
        weight: channel-wise quantization;
        activation: layer-wise quantization(faster than channel-wise);
        
    
    Args Input
        :param fp32_name: the name of FP32 Models
        :param output_dir: the path to INT8 Models
        :param log_dir: the path to logger 
    
    Args Output
        :INT8 Models, saved in ./results/...
    """
    
    # load model 
    model = TinyLIC(model_size = "80M")
    snapshot = torch.load('./pretrained/'+fp32_name+'.pth.tar', map_location=device)['state_dict']
    model.load_state_dict(snapshot, strict=False)
    model.update(force=True)
    model.entropy_bottleneck.update()
    model = model.to(device).eval()
    
    if args.test_before_calibration:
        logging.info('Full-precision model: psnr= {:.2f}; ms-ssim={:.4f}; bpp= {:.3f}'.format(*validate_model(model)))

    
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': True, 'symmetric': False, 'scale_method': 'max'}
    aq_params = {'channel_wise': False, 'symmetric': False, 'scale_method': 'max', 'leaf_param': True}
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.to(device)
    qnn.eval()

    qnn = generator(qnn, args)

    qnn.set_quant_state(weight_quant=True, act_quant=True)

    torch.save(qnn.state_dict(), "{}/INT8.pth".format(output_dir))
    
    return qnn

def init_path(args):
    base_dir = f'./results/' + args.fp32_name + '/'
    output_dir = os.path.join(base_dir, 'outputs')

    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def quantize(fp32_name):
    args.fp32_name = fp32_name
    output_dir = init_path(args)

    seed_all(args.seed)
    qnn = quantize_int8(args, fp32_name, output_dir)
    
    return qnn


            















