import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from compressai.layers.gdn import GDN
from quantization.quantizer import StraightThrough, round_ste, lp_loss, UniformAffineQuantizer


class QuantModule(nn.Module):
    r"""
        Convert module to quantmodule.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.ConvTranspose2d, nn.LayerNorm, nn.Linear, GDN, nn.PixelShuffle], weight_quant_params: dict = {},
                 act_quant_params: dict = {},  disable_act_quant: bool = False, se_module=None):
        super(QuantModule, self).__init__()

        self.if_layer_norm = False
        self.if_tconv = False
        self.is_ps = False
        if isinstance(org_module, nn.Conv2d):
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
            # dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d

        elif isinstance(org_module, nn.ConvTranspose2d):
            # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
            # output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding, 
                                   output_padding=org_module.output_padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv_transpose2d
            self.if_tconv = True

        elif isinstance(org_module, nn.Linear):
            # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear

        elif isinstance(org_module, nn.LayerNorm):
            # torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
            self.fwd_kwargs = dict(normalized_shape = org_module.normalized_shape)
            # self.fwd_kwargs = dict()
            self.fwd_func = F.layer_norm
            self.if_layer_norm = True
        
        elif isinstance(org_module, GDN):
            # torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
            self.fwd_kwargs = dict(inverse = org_module.inverse, gamma_reparam = org_module.gamma_reparam, 
                                   beta_reparam = org_module.beta_reparam)
            # self.fwd_kwargs = dict()
            self.fwd_func = f_gdn
        
        elif isinstance(org_module, nn.PixelShuffle):
            self.fwd_kwargs = org_module.upscale_factor
            self.fwd_func = F.pixel_shuffle
            self.is_ps = True
        
        else:
            raise ValueError('Not supported modules: {}'.format(org_module))
        
        
        if isinstance(org_module, GDN):
            self.weight = org_module.gamma
            self.org_weight = org_module.gamma.data.clone()
            if org_module.beta is not None:
                self.bias = org_module.beta
                self.org_bias = org_module.beta.data.clone()
            else:
                self.bias = None
                self.org_bias = None
        elif self.is_ps:
            self.weight = None
            self.org_weight = None
            self.bias = None
            self.org_bias = None
        else:
            self.weight = org_module.weight
            self.org_weight = org_module.weight.data.clone()
            if org_module.bias is not None:
                self.bias = org_module.bias
                self.org_bias = org_module.bias.data.clone()
            else:
                self.bias = None
                self.org_bias = None

        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant

        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(tconv=self.if_tconv, **weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(tconv=self.if_tconv,**act_quant_params)

        self.activation_function = nn.LeakyReLU(inplace=True) if self.is_ps else StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr
        self.trained = False
    
    def forward(self, input: torch.Tensor):
        if self.is_ps:
            out = self.fwd_func(input, self.fwd_kwargs)
            out = self.activation_function(out)
            return out
        
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        
        if self.if_layer_norm:
            out = self.fwd_func(input, weight=weight, bias=bias, **self.fwd_kwargs)
        else:
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        # out = ActQuantizer(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant and self.trained:
            out = self.act_quantizer(out, True)
        return out
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant



def f_gdn(x, gamma, beta, inverse, gamma_reparam, beta_reparam):
    _, C, _, _ = x.size()
    gamma = gamma_reparam(gamma)
    beta = beta_reparam(beta)
    gamma = gamma.reshape(C, C, 1, 1)
    norm = F.conv2d(x**2, gamma, beta)
    if inverse:
        norm = torch.sqrt(norm)
    else:
        norm = torch.rsqrt(norm)
    
    out = x * norm
    return out
        
        
        
        
        
        
        
        
        
        
        
        
        
