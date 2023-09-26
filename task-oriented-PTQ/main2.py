import os
import sys
import copy
import time
import random
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn

from quantization import *
from ckpts.image import nic_tic
from utils import AverageMeter, setup_logger, init_lic, Logger
from losses.losses import Metrics, RateDistortionLoss
from datasets.dataset import get_dataloader, get_train_samples
from test_datasets import Test_kodak


def parse_args(argv):
    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--resume', type=str, help='snapshot path')
    parser.add_argument('--quality', default=6, type=int, help='model quality from 1-6')
    parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size for data loader')
    parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str, help='result dir name')
    parser.add_argument('--arch', default='Lu2022', type=str, help='model name', 
                        choices=['Lu2022', 'Cheng2020', ])
    parser.add_argument('--type', default='mse', type=str, help='model loss type', choices=['mse','ms-ssim',])
    parser.add_argument('--lmbda', default=0.0483, type=float, 
                        help='the lmbda related to quality, 0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483')
    parser.add_argument('--save', action='store_true', help='save quantized model')

    # quantization parameters
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    #parser.add_argument('--test_before_calibration', action='store_true')
    parser.add_argument('--test_before_calibration', default=True, type=bool, help='test_before_calibration')

    # weight calibration parameters
    parser.add_argument('--input_prob', default=0.5, type=float)
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate')
    parser.add_argument('--task_loss', default=2, type=float, help='task-loss')
    parser.add_argument('--num_samples', default=12, type=int, help='batch_size * n, size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')
    parser.add_argument('--init', default='max', type=str, help='param init type', 
                        choices=['max','mse', 'gaussian', 'l1', 'l2', ])

    args = parser.parse_args(argv)

    if not args.config:
        if args.resume:
            assert args.resume.startswith('./')
            dir_path = '/'.join(args.resume.split('/')[:-2])
            args.config = os.path.join(dir_path, 'config.yaml')
        else:
            args.config = './config.yaml'

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

    
def set_train(model, flag=False):
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            module.trained = flag
        else:
            set_train(module, flag)



def validate_model(logger, test_dataloader, model, criterion, metric):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter('RD Loss', ':6.3f')
    bpp_loss = AverageMeter('Bpp Loss', ':6.3f')
    mse_loss = AverageMeter('MSE Loss', ':6.3f')
    ms_ssim_loss = AverageMeter('MS-SSIM Loss', ':6.3f')
    psnr_sum = 0.0
    msssim_sum = 0.0
    bit_sum = 0.0

    with torch.no_grad():
        for i, x in enumerate(test_dataloader):
            x = x.to(device)
            out_net = model(x)
            out_net['x_hat'].clamp_(0, 1)

            out_criterion = criterion(out_net, x)
            bpp, psnr, ms_ssim = metric(out_net, x)
            psnr_sum += psnr
            msssim_sum += ms_ssim
            bit_sum += bpp

            logger.update_test(bpp, psnr, ms_ssim, out_criterion, model.aux_loss())

        logger.print_test()
        logger.write_test()

        loss.update(logger.loss.avg)
        bpp_loss.update(logger.bpp_loss.avg)
        mse_loss.update(logger.mse_loss.avg)
        ms_ssim_loss.update(logger.ms_ssim_loss.avg)
        logging.info(f'[ Test ] Total mean of loss: {loss.avg:.4f}')
        
    logger.init()
    model.train()

#     print(f'AVG PSNR: {psnr_sum/(i+1):.4f}dB')
#     print(f'AVG MS-SSIM: {msssim_sum/(i+1):.4f}')
#     print(f'AVG Bit-rate: {bit_sum/(i+1):.4f} bpp')

    logging.info(f'[ Test ] AVG PSNR: {psnr_sum/(i+1):.4f}dB')
    logging.info(f'[ Test ] AVG MS-SSIM: {msssim_sum/(i+1):.4f}')
    logging.info(f'[ Test ] AVG Bit-rate: {bit_sum/(i+1):.4f} bpp')

    return loss.avg, bpp_loss.avg, mse_loss.avg, ms_ssim_loss.avg


def optimize_model(args, config, output_dir, log_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader, test_dataloader = get_dataloader(config)
    # criterion = RateDistortionLoss(lmbda=config['lmbda'], metric=config['metric'])
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric=args.type)
    metric = Metrics()
    logger = Logger(config, output_dir, log_dir)

    # load model 
    is_cheng = False
    if 'Lu2022' in args.arch:
        model = nic_tic(args.quality, args.type, pretrained=True)
    elif 'Cheng2020' in args.arch:
        is_cheng = True
        model = torch.load('./ckpts/Cheng2020/{}/cheng2020_{}_{}.pth'.format(args.type, args.type, args.quality))
    elif 'Minnen2018' in args.arch:
        model = torch.load('./ckpts/Minnen2018/{}/minnen2018_{}_{}.pth'.format(args.type, args.type, args.quality))
    else:
        raise ValueError(f'Invalid model arch "{args.arch}"')
    # device = next(model.parameters()).device
    model.to(device)
    model.eval()

    if args.test_before_calibration:
        logging.info('=======================Full-precision model========================')
        Test_kodak(model)
        # print('Full-precision model loss: rd-loss= {}; bpp-loss={}; mse-loss= {}; ms-ssim loss= {}'.format(*validate_model(logger, test_dataloader, model, criterion, metric)))

    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': args.init}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': args.channel_wise, 'scale_method': args.init, 'leaf_param': args.act_quant}
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params, is_cheng = is_cheng)
    qnn.to(device)
    qnn.eval()
    if not args.disable_8bit_head_stem:
        # print('Setting the first and the last layer to 8-bit')
        logging.info('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
    logging.info('quantized model architecture: {}'.format(qnn))

    qnn.disable_network_output_quantization()

    # load calibration dataset
    cali_data = get_train_samples(train_dataloader, num_samples=args.num_samples)
    device = next(model.parameters()).device
    # print(cali_data.shape)

    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)
    # print(cali_data[:args.batch_size].shape)
    logging.info(cali_data[:args.batch_size].shape)
    init_start = time.time()
    _ = qnn(cali_data[:args.batch_size].to(device))
    init_time = time.time() - init_start
    # print('Init time: {}'.format(init_time))
    logging.info('Init time: {}'.format(init_time))

    if args.test_before_calibration:
        set_train(qnn, False)
        qnn.set_quant_state(False, False)
        logging.info('=======================Close quantization model========================')
        Test_kodak(qnn)

        qnn.set_quant_state(True, False)
        logging.info('=======================Weight quantization model w/o opt========================')
        Test_kodak(qnn)
        
        # # torch.save(qnn, './results/{}_LW.pth'.format(args.type))
        # torch.save(qnn, "{}/{}_Q{}_W{}A{}_{}-init_LW.pth".format(output_dir ,args.arch, args.quality, args.n_bits_w, args.n_bits_a, args.init))
        # return 

        # logging.info('Quantized model W{}A32 loss w/o optimization: rd-loss= {}; bpp-loss={}; mse-loss= {}; ms-ssim loss= {}'.format(args.n_bits_w, *validate_model(logger, test_dataloader, qnn, criterion, metric)))
        
        set_train(qnn, False)
    
    
    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, batch_size = args.batch_size, iters=args.iters_w, weight=args.weight, 
                  input_prob=args.input_prob, lr=args.lr, asym=True, b_range=(args.b_start, args.b_end), warmup=args.warmup, 
                  act_quant=args.act_quant, opt_mode='mse', config=config, args=args)

    def recon_model(model: nn.Module):
        """
        layer wise optimization
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    # print('Ignore reconstruction of layer {}'.format(name))
                    logging.info('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    # print('Reconstruction for layer {}'.format(name))
                    logging.info('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, name, **kwargs)
            
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    # print('Ignore reconstruction of block {}'.format(name))
                    logging.info('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    # print('Reconstruction for block {}'.format(name))
                    logging.info('Reconstruction for block {}'.format(name))
                    block_reconstruction(qnn, module, name, **kwargs)

            else:
                recon_model(module)


    # Start calibration
    qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
    if 'Lu2022' in args.arch:
        qnn.model.g_s7.set_quant_state(True, False)
    elif 'Cheng2020' in args.arch:
        qnn.model.g_s[-1][0].set_quant_state(True, False)
    else:
        qnn.model.g_s[-1].set_quant_state(True, False)
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
#     print('Quantized model loss w/ optimization: rd-loss= {}; bpp-loss={}; mse-loss= {}'.format(*validate_model(logger, test_dataloader, qnn, criterion, metric)))
    # logging.info('Quantized model W{}A32 loss w/ optimization: rd-loss= {}; bpp-loss={}; mse-loss= {}; ms-ssim loss= {}'.format(args.n_bits_w, *validate_model(logger, test_dataloader, qnn, criterion, metric)))
    logging.info('=======================Weight quantization model w/ opt========================')
    Test_kodak(qnn.eval())


    qnn.set_quant_state(weight_quant=True, act_quant=True)
    if 'Lu2022' in args.arch:
        qnn.model.g_s7.set_quant_state(True, False)
    elif 'Cheng2020' in args.arch:
        qnn.model.g_s[-1][0].set_quant_state(True, False)
    else:
        qnn.model.g_s[-1].set_quant_state(True, False)

    # logging.info('Quantized model W{}A{} loss w/ optimization: rd-loss= {}; bpp-loss={}; mse-loss= {}; ms-ssim loss= {}'.format(args.n_bits_w, args.n_bits_a, *validate_model(logger, test_dataloader, qnn, criterion, metric)))
    logging.info('=======================Fully quantization model w/ opt========================')
    Test_kodak(qnn.eval())

    
    if args.save:
        logging.info('save quantized model in {}'.format(output_dir))
        if args.channel_wise:
            torch.save(qnn, "{}/{}_Q{}_W{}A{}_prob{}_task{}_{}-init_{}_CW.pth".format(output_dir ,args.arch, args.quality, args.n_bits_w, args.n_bits_a, args.input_prob, args.task_loss, args.init, config['c_data']))
        else:
            torch.save(qnn, "{}/{}_Q{}_W{}A{}_prob{}_task{}_{}-init_{}_LW.pth".format(output_dir ,args.arch, args.quality, args.n_bits_w, args.n_bits_a, args.input_prob, args.task_loss, args.init, config['c_data']))


def main(argv):
    args = parse_args(argv)
    config, output_dir, log_dir = init_lic(args)
    #config = get_config(args.config)

    seed_all(args.seed)
    setup_logger(log_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')

    logging.info('[PID] %s'%os.getpid())
    msg = f'======================= Hyper Parameters ======================='
    logging.info(msg)
    logging.info('task loss: {}'.format(args.task_loss))
    logging.info('calib data: {}'.format(config['c_data']))
    logging.info('param init: {}'.format(args.init))
    logging.info('channel wise: {}'.format(args.channel_wise))
    
    logging.info('seed: {}'.format(args.seed))
    logging.info('lambda: {}'.format(args.lmbda))
    logging.info('iterations: {}'.format(args.iters_w))
    logging.info('batch_size: {}'.format(args.batch_size))
    logging.info('loss weight: {}'.format(args.weight))
    logging.info('input drop rate: {}'.format(args.input_prob))
    end = f'========================== {args.arch} =========================='
    logging.info(end)
    
    optimize_model(args, config, output_dir, log_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
