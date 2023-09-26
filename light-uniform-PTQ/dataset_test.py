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
from quant import quantize


device = 'cuda'
warnings.filterwarnings("ignore")

def parse_args(argv):
    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--save', default=True, help='save quantized model')
    parser.add_argument('--fp32_name',default='tinylic', help='fp32_model_path')
    parser.add_argument('--type',default='INT8', help='INT8 or FP16 or FP32', choices=['INT8','FP16','FP32'])
    parser.add_argument('--gt_path',default='./data/Kodak', help='the path to ground truth images folder')
    parser.add_argument('--lrd',default=0.0004, type=float, help='the parameter to control bit rate')
    parser.add_argument('--freq',default=5, type=int, help='the frequence of print')


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

    
def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

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

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

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
    
       
    
def main_fp32(gt_path, model_path, lrd, freq=100):
    """
    Test FP32 Models in dataset;
    
    Args Input
        :param gt_path: the path to datasets
        :param model_path: the name of FP32 Models
        :param lrd: the lambda to control bit-rate, suggest [0.00005, 0.0009]
        :param freq: the frequence of print
    
    Args Output
        :reconstruction image, saved in ./results/...
    """
    os.makedirs('./results/'+model_path+'/outputs/FP32',exist_ok=True)
    
    # load models
    net = TinyLIC()
    snapshot = torch.load("./pretrained/"+model_path+ ".pth.tar", map_location=device)['state_dict']
    net.load_state_dict(snapshot, strict=False)
    net.update(force=True)
    net = net.to(device).eval()
    torch.save({'state_dict': net.state_dict()}, './results/'+model_path+'/outputs/FP32.pth')
    model_size = filesize('./results/'+model_path+'/outputs/FP32.pth')
    
    # metrics
    sum_psnr = 0.0
    sum_msssim = 0.0
    sum_byte = 0.
    sum_bpp = 0.0
    sum_enc = 0.
    sum_dec = 0.
    
    i = 0
    lambda_rd = torch.tensor([lrd]).to(device)
    

    for img_name in (os.listdir(gt_path)):
        
        base_name, _ = os.path.splitext(img_name)
        base_name = os.path.basename(base_name)
        
        img = Image.open(gt_path+'/'+img_name).convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        H, W = x.size(2), x.size(3)
        if H > W:
            x = F.interpolate(x, (240, 180), mode = 'area').clamp(0,1)
        else:
            x = F.interpolate(x, (180, 240), mode = 'area').clamp(0,1)
        
        gt_down = transforms.ToPILImage()(x.squeeze().cpu())
        
        p = 64
        h, w = x.size(2), x.size(3)
        x_pad = pad(x, p)
        
        with torch.no_grad():
                    
            # compress
            torch.cuda.synchronize()
            start_compress = time.time()
            out_enc = net.compress(x_pad, lambda_rd)
            torch.cuda.synchronize()
            end_compress = time.time()

            # decompress
            torch.cuda.synchronize()
            start_decompress = time.time()
            out = net.decompress(out_enc["strings"], out_enc["shape"], lambda_rd)
            torch.cuda.synchronize()
            end_decompress = time.time()

            # save
            rec = crop(out['x_hat'], (h, w))            
        
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        byte = sum(len(s[0]) for s in out_enc["strings"])
        bpp = byte * 8.0 / num_pixels
        psnr = compute_psnr(x, rec)
        msssim = compute_msssim(x, rec)
        enc_time = end_compress - start_compress
        dec_time = end_decompress - start_decompress
        
        if i%freq == 0 or psnr<15.:
            print(img_name)
            print(i)
            print(f'PSNR: {psnr:.2f}dB')
            print(f'MS-SSIM: {msssim:.4f}')
            print(f'Byte: {byte:.0f} Byte')
            print(f'Bit-rate: {bpp:.3f} bpp')
            gt_down.save('./results/'+model_path+'/outputs/FP32/'+img_name, format="PNG")
            rec_tic = transforms.ToPILImage()(rec.squeeze().cpu())
            rec_tic.save('./results/'+model_path+'/outputs/FP32/rec_'+img_name, format="PNG")
            print("::::::::::::::::::::::::::::::::")
            
        sum_psnr += psnr
        sum_msssim += msssim
        sum_byte += byte
        sum_bpp += bpp
        sum_enc += enc_time
        sum_dec += dec_time
        i += 1
    
    print("============ Average Results ===============")
    print(f'Data Length: {len(os.listdir(gt_path))}')
    print(f'AVG PSNR: {(sum_psnr/len(os.listdir(gt_path))):.2f}dB')
    print(f'AVG MS-SSIM: {(sum_msssim/len(os.listdir(gt_path))):.4f}')
    print(f'AVG Byte: {(sum_byte/len(os.listdir(gt_path))):.3f} Byte')
    print(f'AVG Bit-rate: {(sum_bpp/len(os.listdir(gt_path))):.3f} bpp')
    print(f'AVG Enc Time: {(sum_enc/len(os.listdir(gt_path))):.3f} s')
    print(f'AVG Dec Time: {(sum_dec/len(os.listdir(gt_path))):.3f} s')
    print(f'Model Size: {model_size/1024**2:.2f}MB')


def main_fp16(gt_path, model_path, lrd, freq=100):
    """
    Test FP16 Models in dataset;
    
    Args Input
        :param gt_path: the path to datasets
        :param model_path: the name of FP16 Models
        :param lrd: the lambda to control bit-rate, suggest [0.00005, 0.0009]
        :param freq: the frequence of print
    
    Args Output
        :reconstruction image, saved in ./results/...
    """
    os.makedirs('./results/'+model_path+'/outputs/FP16',exist_ok=True)
    
    # load models
    net = torch.load("./results/"+model_path+ "/outputs/FP16.pth", map_location=device)
    net = net.to(device).eval()
    model_size = filesize("./results/"+model_path+ "/outputs/FP16.pth")
    
    # metrics
    sum_psnr = 0.0
    sum_msssim = 0.0
    sum_byte = 0.
    sum_bpp = 0.0
    sum_enc = 0.
    sum_dec = 0.
    
    i = 0
    lambda_rd = torch.tensor([lrd]).to(device).half()
    
    
    for img_name in (os.listdir(gt_path)):
        
        base_name, _ = os.path.splitext(img_name)
        base_name = os.path.basename(base_name)
        
        img = Image.open(gt_path+'/'+img_name).convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device).half()
        H, W = x.size(2), x.size(3)
        if H > W:
            x = F.interpolate(x, (240, 180), mode = 'area').clamp(0,1)
        else:
            x = F.interpolate(x, (180, 240), mode = 'area').clamp(0,1)
        
        gt_down = transforms.ToPILImage()(x.squeeze().cpu())
        
        p = 64
        h, w = x.size(2), x.size(3)
        x_pad = pad(x, p)
        
        with torch.no_grad():
            
        
            # compress
            torch.cuda.synchronize()
            start_compress = time.time()
            out_enc = net.compress(x_pad, lambda_rd)
            torch.cuda.synchronize()
            end_compress = time.time()

            # decompress
            torch.cuda.synchronize()
            start_decompress = time.time()
            out = net.decompress(out_enc["strings"], out_enc["shape"], lambda_rd)
            torch.cuda.synchronize()
            end_decompress = time.time()

            # save
            rec = crop(out['x_hat'], (h, w))            
        
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        byte = sum(len(s[0]) for s in out_enc["strings"])
        bpp = byte * 8.0 / num_pixels
        psnr = compute_psnr(x, rec)
        msssim = compute_msssim(x, rec)
        enc_time = end_compress - start_compress
        dec_time = end_decompress - start_decompress
        
        if i%freq == 0 or psnr<15.:
            print(img_name)
            print(i)
            print(f'PSNR: {psnr:.2f}dB')
            print(f'MS-SSIM: {msssim:.4f}')
            print(f'Byte: {byte:.0f} Byte')
            print(f'Bit-rate: {bpp:.3f} bpp')
            gt_down.save('./results/'+model_path+'/outputs/FP16/'+img_name, format="PNG")
            rec_tic = transforms.ToPILImage()(rec.squeeze().cpu())
            rec_tic.save('./results/'+model_path+'/outputs/FP16/rec_'+img_name, format="PNG")
            print("::::::::::::::::::::::::::::::::")
            
        sum_psnr += psnr
        sum_msssim += msssim
        sum_byte += byte
        sum_bpp += bpp
        sum_enc += enc_time
        sum_dec += dec_time
        i += 1
    
    print("============ Average Results ===============")
    print(f'Data Length: {len(os.listdir(gt_path))}')
    print(f'AVG PSNR: {(sum_psnr/len(os.listdir(gt_path))):.2f}dB')
    print(f'AVG MS-SSIM: {(sum_msssim/len(os.listdir(gt_path))):.4f}')
    print(f'AVG Byte: {(sum_byte/len(os.listdir(gt_path))):.3f} Byte')
    print(f'AVG Bit-rate: {(sum_bpp/len(os.listdir(gt_path))):.3f} bpp')
    print(f'AVG Enc Time: {(sum_enc/len(os.listdir(gt_path))):.3f} s')
    print(f'AVG Dec Time: {(sum_dec/len(os.listdir(gt_path))):.3f} s')
    print(f'Model Size: {model_size/1024**2:.2f}MB')


            
def main_int8(gt_path, model_path, net, lrd, freq=100):
    """
    Test INT8 Models in dataset;
    
    Args Input
        :param gt_path: the path to datasets
        :param model_path: the name of INT8 Models
        :param lrd: the lambda to control bit-rate, suggest [0.00005, 0.0009]
        :param freq: the frequence of print
    
    Args Output
        :reconstruction image, saved in ./results/...
    """
    os.makedirs('./results/'+model_path+'/outputs/INT8',exist_ok=True)
    
    # load models
    model_size = filesize("./results/"+model_path+ "/outputs/INT8.pth")
    
    # metrics
    sum_psnr = 0.0
    sum_msssim = 0.0
    sum_byte = 0.
    sum_bpp = 0.0
    sum_enc = 0.
    sum_dec = 0.
    
    i = 0
    lambda_rd = torch.tensor([lrd]).to(device)
    
    
    for img_name in (os.listdir(gt_path)):
        
        base_name, _ = os.path.splitext(img_name)
        base_name = os.path.basename(base_name)
        
        img = Image.open(gt_path+'/'+img_name).convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        H, W = x.size(2), x.size(3)
        if H > W:
            x = F.interpolate(x, (240, 180), mode = 'area').clamp(0,1)
        else:
            x = F.interpolate(x, (180, 240), mode = 'area').clamp(0,1)
        
        gt_down = transforms.ToPILImage()(x.squeeze().cpu())
        
        p = 64
        h, w = x.size(2), x.size(3)
        x_pad = pad(x, p)
        
        with torch.no_grad():
            
        
            # compress
            torch.cuda.synchronize()
            start_compress = time.time()
            out_enc = net.compress(x_pad, lambda_rd)
            torch.cuda.synchronize()
            end_compress = time.time()

            # decompress
            torch.cuda.synchronize()
            start_decompress = time.time()
            out = net.decompress(out_enc["strings"], out_enc["shape"], lambda_rd)
            torch.cuda.synchronize()
            end_decompress = time.time()

            # save
            rec = crop(out['x_hat'], (h, w))            
        
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        byte = sum(len(s[0]) for s in out_enc["strings"])
        bpp = byte * 8.0 / num_pixels
        psnr = compute_psnr(x, rec)
        msssim = compute_msssim(x, rec)
        enc_time = end_compress - start_compress
        dec_time = end_decompress - start_decompress
        
        if i%freq == 0 or psnr<15.:
            print(img_name)
            print(i)
            print(f'PSNR: {psnr:.2f}dB')
            print(f'MS-SSIM: {msssim:.4f}')
            print(f'Byte: {byte:.0f} Byte')
            print(f'Bit-rate: {bpp:.3f} bpp')
            gt_down.save('./results/'+model_path+'/outputs/INT8/'+img_name, format="PNG")
            rec_tic = transforms.ToPILImage()(rec.squeeze().cpu())
            rec_tic.save('./results/'+model_path+'/outputs/INT8/rec_'+img_name, format="PNG")
            print("::::::::::::::::::::::::::::::::")
            
        sum_psnr += psnr
        sum_msssim += msssim
        sum_byte += byte
        sum_bpp += bpp
        sum_enc += enc_time
        sum_dec += dec_time
        i += 1
    
    print("============ Average Results ===============")
    print(f'Data Length: {len(os.listdir(gt_path))}')
    print(f'AVG PSNR: {(sum_psnr/len(os.listdir(gt_path))):.2f}dB')
    print(f'AVG MS-SSIM: {(sum_msssim/len(os.listdir(gt_path))):.4f}')
    print(f'AVG Byte: {(sum_byte/len(os.listdir(gt_path))):.3f} Byte')
    print(f'AVG Bit-rate: {(sum_bpp/len(os.listdir(gt_path))):.3f} bpp')
    print(f'AVG Enc Time: {(sum_enc/len(os.listdir(gt_path))):.3f} s')
    print(f'AVG Dec Time: {(sum_dec/len(os.listdir(gt_path))):.3f} s')
    print(f'Model Size: {model_size/1024**2:.2f}MB')

                   
    
def main(argv):
    args = parse_args(argv)
    seed_all(args.seed)
    
    if args.type == 'FP32':
        main_fp32(args.gt_path, args.fp32_name, args.lrd, args.freq)
    elif args.type == 'FP16':
        main_fp16(args.gt_path, args.fp32_name, args.lrd, args.freq)
    elif args.type == 'INT8':
        net = quantize(args.fp32_name)
        main_int8(args.gt_path, args.fp32_name, net, args.lrd, args.freq)
    else:
        raise NotImplementedError
        

if __name__ == '__main__':
    main(sys.argv[1:])   
    
    
    
    
    
    
    
    
    
    
    
    
    