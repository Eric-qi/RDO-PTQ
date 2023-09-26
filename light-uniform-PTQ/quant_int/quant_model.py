import torch.nn as nn

from quant_int.fold_bn import search_fold_and_remove_bn
from quant_int.quant_layer import QuantModule, StraightThrough
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, is_fusing=False):
        super().__init__()
        if is_fusing:
            search_fold_and_remove_bn(model)
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        else:
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
            
            # elif isinstance(child_module, (RSTB)):
            #     setattr(module, name, child_module)
            
            elif isinstance(child_module, (nn.LeakyReLU, nn.GELU, nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue
                
            elif isinstance(child_module, StraightThrough):
                continue
            
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input, lamda):
        return self.model(input, lamda)
    
    def compress(self, *input):
        return self.model.compress(*input)
    
    def decompress(self, input, shape, *params):
        return self.model.decompress(input, shape, *params)
    
    def compress_split(self, *input):
        return self.model.compress_split(*input)
    
    def decompress_split(self, input, shape, *params):
        return self.model.decompress(input, shape, *params)
    
    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss


    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True
