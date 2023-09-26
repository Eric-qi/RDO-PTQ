import time
import torch
import logging
import torch.nn.functional as F

from quantization.quant_layer import QuantModule
from quantization.quant_model import QuantModel
from quantization.quant_block import BaseQuantBlock
from quantization.quantizer import AdaRoundQuantizer, lp_loss,  StraightThrough , round_ste
from quantization.utils import LinearTempDecay, save_grad_data, save_inp_oup_data, varify_hook

from losses.losses import RateDistortionLoss


def set_mode(model, act_quant):

    for name, module in model.named_children():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if module.trained:
                module.set_quant_state(True, act_quant)
        else:
            set_mode(module, act_quant)

def find_unquantized_module(model: torch.nn.Module, _name_: str = "g_a", module_list: list = [], name_list: list = []):
    r"""
        Only to list unquantized module in current coder.

    Args:
        g_a: main encoder
        h_a: hyper encoder
        g_s: main decoder
        h_s: hyper decoder
    """
    for name, module in model.named_children():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if not module.trained:
                module.set_quant_state(False,False)
                if "g_a" in _name_ and "g_a" in name:
                    name_list.append(name)
                    module_list.append(module)
                if "h_a" in _name_ and "h_a" in name:
                    name_list.append(name)
                    module_list.append(module)
                if "h_s" in _name_ and "h_s" in name:
                    name_list.append(name)
                    module_list.append(module)
                if "g_s" in _name_ and "g_s" in name:
                    name_list.append(name)
                    module_list.append(module)            
        else:
            find_unquantized_module(module, _name_, module_list, name_list)
    return module_list[1:], name_list[1:]            

def fp_out(model: torch.nn.Module, _name_: str = "g_a", module_list: list = [], in_quant: torch.Tensor = None):
    r"""
        Simplified task output to reduce training time. 

    Args:
        module_list (list): subsequent unquantized modules
        in_quant (tensor): the quantized input of current layer
    
    Return:
        g_a: the latent representation of main encoder
        h_a: the latent representation of hyper encoder
        h_s: the latent representation of hyper decoder
        g_s: the reconstructed representation of main decoder
        others: current output
    """
    output = in_quant
    for num, module in enumerate(module_list):
        if isinstance(module, (QuantModule)):
            output = module(output)
        else:
            x_size = output.shape[2:4]
            output = module(output, (x_size[0], x_size[1]))
    if "g_a" in _name_:
        # y_hat = model.model.gaussian_conditional.quantize(output,"")
        y_hat = round_ste(output)
        return y_hat
#     elif "h_a" in _name_:
#         z_hat, z_likelihoods = model.model.entropy_bottleneck(output)
#         return model.model.h_s(z_hat)
    else:
        return output

class LossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 lmbda=None,
                 metric=None):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lmbda = lmbda
        self.metric = metric

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, quant_net_out=None, cali_data=None, grad=None):
        r"""
            Compute the total loss for optimization:  
                rec_loss is the output reconstruction loss of current layer, 
                round_loss is a regularization term to optimize the rounding policy,
                task_loss is the output reconstruction loss of current coder.

        Args:
            pred (tensor): output from current quantized layer
            tgt (tensor): the floating-point output of current layer
            quant_net_out (tensor): output from current quantized coder
            cali_data (tensor): the floating-point output of current coder
            grad (tensor): gradients to compute fisher information
            return: total loss function
        """
        self.count += 1


        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))
        
        task_loss = 0.
        if quant_net_out is not None:
            # criterion = RateDistortionLoss(lmbda=self.lmbda, metric=self.metric)
            # out_criterion = criterion(quant_net_out, cali_data)
            # task_loss = out_criterion['loss']
            if self.rec_loss == 'mse':
                task_loss = lp_loss(quant_net_out, cali_data, p=self.metric)
            elif self.rec_loss == 'fisher_diag':
                return 
            elif self.rec_loss == 'fisher_full':
                return
            else:
                raise ValueError('Not supported rtask loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = round_loss + rec_loss + task_loss
        if self.count % 500 == 0:
            logging.info('Total loss:\t{:.3f} ( task:{:.3f}, rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(task_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


def block_reconstruction(model: QuantModel, block: BaseQuantBlock, block_name: str , cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, input_prob: float = 1.0, 
                         act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         config=None, args=None):
    r"""
        RSTB, Residual Swin Transformer Block, optimization.

    Args:
            model: QuantModel
            block: RSTB
            block_name: current RSTB name
            cali_data: data for calibration, typically k·batch_size for simplicity (e.g., 3·4 = 12)

            batch_size: mini-batch size
            iters: optimization iterations
            weight: the weight of rounding regularization term
            opt_mode: optimization mode

            asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
            include_act_func: optimize the output after activation function
            b_range: temperature range

            warmup: proportion of iterations that no scheduling for temperature
            input_prob: drop strategy proposed in QDrop, to improve generalization (0-1)

            act_quant: use activation quantization or not.
            lr: learning rate
            p: L_p norm minimization

            config: config file
            args: main.py args
    """

    device = 'cuda'
    cached_start = time.time()
    cached_inps, cached_outs = save_inp_oup_data(model, block, cali_data, asym, act_quant, batch_size=1, input_prob=True)
    cached_time = time.time() - cached_start
    logging.info('Cached init time: {}'.format(cached_time))
    
    module_list, name_list = [], []
    module_list, name_list = find_unquantized_module(model, block_name, module_list, name_list)
    logging.info(name_list)
    
    
    model.set_quant_state(False, False)
    with torch.no_grad():
        fp_net_out = fp_out(model, block_name, module_list, cached_outs.to(device))
     
    set_mode(model, act_quant)
    block.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'
     
#     for name, module in block.named_modules():
#         if isinstance(module, (QuantModule, BaseQuantBlock)):
#             module.trained = True

    if not include_act_func:
        org_act_func = block.activation_function
        block.activation_function = StraightThrough()


    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                        weight_tensor=module.org_weight.data)
            module.weight_quantizer.soft_targets = True


    opt_params = []
    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            opt_params += [module.weight_quantizer.alpha]
    optimizer = torch.optim.Adam(opt_params)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    #     #Use UniformAffineQuantizer to learn delta
    #     if hasattr(block.act_quantizer, 'delta') and block.act_quantizer.delta is not None:
    #         opt_params = [block.act_quantizer.delta]
    #     else:
    #         opt_params = []
    #     for name, module in block.named_modules():
    #         if isinstance(module, (QuantModule)):
    #             module.act_quantizer.act = True
    #             if module.act_quantizer.delta is not None:
    #                 opt_params += [module.act_quantizer.delta]
    #                 module.act_quantizer.is_training = True
    #     optimizer = torch.optim.Adam(opt_params, lr=lr)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    # loss_mode = 'none' if act_quant else 'relaxation'
    loss_mode = 'relaxation'
    rec_loss = opt_mode

    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight, 
                             max_count=iters, rec_loss=rec_loss, b_range=b_range, 
                             decay_start=0, warmup=warmup, p=p, lmbda=args.lmbda, metric=args.task_loss)

    

    # varify the data
    # if not varify_hook:
    #     raise ValueError('register forward hook is error')
    
    if opt_mode != 'mse':
        cached_grads = save_grad_data(model, block, cali_data, act_quant, batch_size=1)
    else:
        cached_grads = None
    device = next(model.parameters()).device
    for i in range(iters):
        idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
        cur_inp, cur_sym = cached_inps[0][idx].to(device), cached_inps[1][idx].to(device)
        if input_prob < 1.0:
            cur_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx].to(device) if opt_mode != 'mse' else None

        optimizer.zero_grad()

        x_size = cur_inp.shape[2:4]
        
        out_quant = block(cur_inp, (x_size[0], x_size[1])) if 'Lu2022' in args.arch else block(cur_inp)
        in_quant = out_quant.to(device)
        
        quant_net_out = None
        quant_net_out = fp_out(model, block_name, module_list, in_quant)
        

        err = loss_func(pred=out_quant, tgt=cur_out, quant_net_out=quant_net_out, cali_data=fp_net_out[idx].to(device), grad=cur_grad)
        err.backward(retain_graph=True)

        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()


    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.act_quantizer.is_training = False
            module.trained = True

    if not include_act_func:
        block.activation_function = org_act_func