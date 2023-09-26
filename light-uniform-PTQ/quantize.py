import os
import sys
import math
import time
import pickle
import random
import warnings
import argparse
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

def parse_args(argv):
    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str, help='result dir name')
    parser.add_argument('--save', default=True, help='save quantized model')
    parser.add_argument('--fp32_name',default='tinylic', help='fp32_model_path')
    parser.add_argument('--type',default='INT8', help='INT8 or FP16', choices=['INT8','FP16',])

    # quantization parameters
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=16, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', default=True, help='apply activation quantization')
    parser.add_argument('--test_before_calibration', default=True, type=bool, help='test_before_calibration')
    parser.add_argument('--sym', default=True, help='symmetric reconstruction')

    args = parser.parse_args(argv)
    
    return args

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
    init_start = time.time()
    _ = qnn(x_pad, lambda_rd)
    init_time = time.time() - init_start
    logging.info('generate quantized model time: {}'.format(init_time))


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
    logging.info('INT8: psnr= {:.2f}; ms-ssim={:.4f}; bpp= {:.3f}'.format(*validate_model(qnn)))
    
    if args.save:
        logging.info('save quantized model in {}'.format(output_dir))
        torch.save(qnn.state_dict(), "{}/INT8.pth".format(output_dir))
    
    return qnn


def quantize_fp16(args, fp32_name, output_dir):
    """
    Quantize a FP32 Models to FP16 Models;
    Based on torch.half to convert;
    
    Args Input
    :param fp32_name: the name of FP32 Models
    
    Args Output
    :FP16 Models, saved in ./results/...
    """
    
    # load FP32 Models and update 
    model = TinyLIC()
    snapshot = torch.load('./pretrained/'+fp32_name+'.pth.tar', map_location=device)['state_dict']
    model.load_state_dict(snapshot, strict=False)
    model.update(force=True)
    
    # quantize to FP16
    model = model.half().to(device).eval()
    
    if args.save:
        logging.info('save quantized model in {}'.format(output_dir))
        # save model, saving state_dict and full_model are both acceptable.
        # torch.save({'state_dict': model.state_dict()}, fp16_dir)
        torch.save(model, "{}/FP16.pth".format(output_dir))


def main(argv):
    args = parse_args(argv)
    output_dir, log_dir = init_lic(args)

    seed_all(args.seed)
    setup_logger(log_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')

    logging.info('[PID] %s'%os.getpid())
    msg = f'======================= {args.fp32_name} ======================='
    logging.info(msg)
    
    if args.type == 'INT8':
        quantize_int8(args, args.fp32_name, output_dir)
    elif args.type == 'FP16':
        quantize_fp16(args, args.fp32_name, output_dir)
    else:
        raise NotImplementedError
        

if __name__ == '__main__':
    main(sys.argv[1:])
            















