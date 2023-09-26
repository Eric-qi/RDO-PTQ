import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from quant_int.quantizer import StraightThrough, round_ste, lp_loss, UniformAffineQuantizer


class QuantModule(nn.Module):


    def __init__(self, org_module: Union[nn.Conv2d, nn.ConvTranspose2d, nn.LayerNorm, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {},  disable_act_quant: bool = False, se_module=None):
        super(QuantModule, self).__init__()

        self.if_layer_norm = False
        self.if_tconv = False
        self.if_conv = False
        self.if_linear = False
        self.org_module = org_module
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
        
        else:
            raise ValueError('Not supported modules: {}'.format(org_module))

        self.weight = org_module.weight
        # self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            # self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            # self.org_bias = None

        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant

        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(tconv=self.if_tconv, **weight_quant_params)
        self.bias_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(tconv=self.if_tconv,**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr
        self.trained = False
    
    def forward(self, input: torch.Tensor):
        
        if not self.weight_quantizer.channel_wise:
            if not self.trained:
                w_int, s_w = self.weight_quantizer(self.weight)
                b_int, s_b = self.bias_quantizer(self.bias)
                r"""
                    b_int = (b_int - self.bias_quantizer.zero_point)/s_w
                    bias rescaling, this version introduce s_w only.
                    you can also introduce s_x.
                """
                b_int = (b_int - self.bias_quantizer.zero_point)/s_w
                b_int = torch.round(b_int*s_b)
                # self.weight.data = w_int
                # print(w_int.max()-w_int.min())
                # print(b_int.max()-b_int.min())
                self.weight.data = torch.tensor(w_int, dtype=torch.uint8)
                # self.bias.data = b_int
                self.bias.data = torch.tensor(b_int, dtype=torch.int16)
                self.trained = True
            
            w_int = self.weight.type_as(input)
            w_int = w_int - self.weight_quantizer.zero_point
            # w_int = self.weight.type_as(input)
            b_int = self.bias.type_as(input)

            if self.if_layer_norm:
                out = self.fwd_func(input, weight=w_int, bias=b_int, **self.fwd_kwargs)
            else:
                out = self.fwd_func(input, w_int, b_int, **self.fwd_kwargs)
            # disable act quantization is designed for convolution before elemental-wise operation,
            # in that case, we apply activation function and quantization after ele-wise op.
            if self.se_module is not None:
                out = self.se_module(out)
            
            out = out*self.weight_quantizer.delta
        else:
            if not self.trained:
                w_int, s_w = self.weight_quantizer(self.weight)
                self.weight.data = torch.tensor(w_int, dtype=torch.uint8)
                self.trained = True
            
            w_int = self.weight.type_as(input)
            w_int = (w_int - self.weight_quantizer.zero_point) * self.weight_quantizer.delta
            b_int = self.bias.type_as(input)
            
            if self.if_layer_norm:
                out = self.fwd_func(input, weight=w_int, bias=b_int, **self.fwd_kwargs)
            else:
                out = self.fwd_func(input, w_int, b_int, **self.fwd_kwargs)
            
        out = self.activation_function(out)
        
        # out = ActQuantizer(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out, True)
        return out
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


