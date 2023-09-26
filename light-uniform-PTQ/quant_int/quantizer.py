import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import warnings
from typing import Any, Callable, List, Optional, Tuple, Union


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

def round_noise(
    self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
) -> Tensor:

    """
    Implement Straight-Through Estimator for rounding noise.
    """
    if mode not in ("noise", "dequantize", "symbols"):
        raise ValueError(f'Invalid quantization mode: "{mode}"')

    if mode == "noise":
        half = float(0.5)
        noise = torch.empty_like(inputs).uniform_(-half, half)
        inputs = inputs + noise
        return inputs

    outputs = inputs.clone()
    if means is not None:
        outputs -= means

    outputs = torch.round(outputs)

    if mode == "dequantize":
        if means is not None:
            outputs += means
        return outputs

    assert mode == "symbols", mode
    outputs = outputs.int()
    return outputs

class round_noise_ste(torch.autograd.Function):
    """
    Implement Straight-Through Estimator for rounding noise.
    """
    @staticmethod
    def forward(ctx, x):
        half = float(0.5)
        noise = torch.empty_like(x).uniform_(-half, half)
        return torch.round(x + noise)

    @staticmethod
    def backward(ctx, g):
        return g

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt):
    """
    loss function measured in L_p Norm
    """

    return (pred-tgt).abs().mean()

def Handle_Parameter(param, b_w=8):
    
    eps = torch.tensor(1e-6, dtype=torch.float32)
    bit_range = 2 ** b_w - 1
    zero_point = param.min()
    param_new = param - zero_point
        
    range_float = param_new.abs().max()
    range_int01 = range_float
    range_int01 = torch.max(range_float, eps)

    param_01 = torch.clamp(param_new / range_int01, -1, 1)
    param_int = torch.round(param_01 * bit_range)
    param_fixed = (param_int / bit_range) * range_int01 + zero_point

    return param_fixed      

        
def ActQuant(x: torch.Tensor):
    x_clone = x.clone().detach()

    if len(x_clone.shape) == 4: # conv
        for i in range(x_clone.shape[1]):
            x_clone[:,i,:,:] = Handle_Parameter(x_clone[:,i,:,:])

    elif len(x_clone.shape) == 3: # linear
        for i in range(x_clone.shape[2]):
            x_clone[:,:,i] = Handle_Parameter(x_clone[:,:,i])

    elif len(x.shape) == 2: # layer_norm
        for i in range(x_clone.shape[1]):
            x_clone[:,i] = Handle_Parameter(x_clone[:,i])

    else: 
        x_clone = Handle_Parameter(x_clone)
    
    return x_clone

# def ActQuantizer(x: torch.Tensor):
#     for i in range(x.shape[0]):
#         x[i] = ActQuant(x[i])
#     return x

def ActQuantizer(x, a_l=8, a_r=8):
    a_low = -2 ** (a_l - 1)
    a_high = 2 ** (a_l - 1)
    a_mult = 2 ** a_r

    out = torch.clamp(x, a_low, a_high)
    out = torch.round(out * a_mult) / a_mult

    return out
        
class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, tconv: bool = False, act: bool = False, prob: float = 1.0):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = torch.empty(0)#None #scaling factors
        self.zero_point = torch.empty(0)#None
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.tconv = tconv
        self.act = act
        
        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    def forward(self, x: torch.Tensor, act: bool = False):
        
        if self.inited is False:
            if self.leaf_param:
                # delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
                # # self.delta = torch.nn.Parameter(delta)
                # # self.zero_point = torch.nn.Parameter(zero_point)
                # self.delta = delta
                # self.zero_point = zero_point

                # self.inited = True
                # x_int = round_ste(x / self.delta) + self.zero_point
                # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                # x_dequant = (x_quant - self.zero_point) * self.delta

                # return x_dequant
                return ActQuantizer(x, a_l=8, a_r=8)


            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                
            self.inited = True

        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        # x_dequant = (x_quant - self.zero_point) * self.delta
        
           
        # x_dequant = (x_quant - self.zero_point)
        # x.data = torch.tensor(x_dequant, dtype=torch.int8)
        return x_quant, self.delta
        
#         if self.leaf_param:
#             return x_dequant
#         else:
#             x.data = x_dequant
#             return x
    

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = torch.empty(0), torch.empty(0)
        if channel_wise:
            # raise NotImplementedError
            x_clone = x.clone().detach()
            if self.tconv:
                n_channels = x_clone.shape[1]
            else:
                n_channels = x_clone.shape[0] # the first dim for conv without considering Tconv

            if len(x.shape) == 4: # conv
                if self.tconv:
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=0)[0] # TODO:Tconv
                else:
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                
            elif len(x.shape) == 1: #layernorm
                x_max = x_clone.abs().max()

            else: # linear ...
                x_max = x_clone.abs().max(dim=-1)[0]

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            if len(x.shape) == 1:
                delta, zero_point = self.init_quantization_scale(x_clone, channel_wise=False)
            else:
                if self.tconv:
                    for c in range(n_channels):
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:, c], channel_wise=False)
                else:
                    for c in range(n_channels):
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)

            if len(x.shape) == 4:
                if self.tconv:
                    delta = delta.view(1, -1, 1, 1)
                    zero_point = zero_point.view(1, -1, 1, 1)
                else:
                    delta = delta.view(-1, 1, 1, 1)
                    zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 1:
                delta = delta.view(-1)
                zero_point = zero_point.view(-1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                # delta = float(x_max - x_min) / (self.n_levels - 1)
                delta = torch.tensor((x_max - x_min) / (self.n_levels - 1))

                delta = torch.max(delta, self.eps)
                
                zero_point = torch.tensor(-x_min / delta)
                zero_point = torch.round(zero_point)
                zero_point = torch.tensor(zero_point).type_as(x)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q)
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        delta = torch.max(delta, self.eps)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        delta = torch.max(delta, self.eps)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)