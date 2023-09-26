import torch.nn as nn

from compressai.layers.gdn import GDN
from quantization.fold_bn import search_fold_and_remove_bn
from quantization.quant_block import specials, BaseQuantBlock, QuantSC
from quantization.quant_layer import QuantModule, StraightThrough
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, is_fusing=True,
                 is_cheng=False):
        super().__init__()
        if is_fusing:
            search_fold_and_remove_bn(model)
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params, is_cheng)
        else:
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params, is_cheng)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, 
                              is_cheng=False):
        r"""
            Recursively replace the module to QuantModule.
        Args:
            module: nn.Module with children modules
            weight_quant_params: quantization parameters for weight quantizer
            act_quant_params: quantization parameters for activation quantizer
        """

        prev_quantmodule = None
        for name, child_module in module.named_children():
            
                
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
                # self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)
            
#             elif 'h_s' in name and is_cheng:
#                 # setattr(module, str(2), QuantSC(child_module[2], weight_quant_params, act_quant_params))
#                 continue

            elif isinstance(child_module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm, GDN, 
                                          nn.PixelShuffle)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
            
            
            elif isinstance(child_module, (nn.LeakyReLU, nn.GELU, nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue
                
            elif isinstance(child_module, StraightThrough):
                continue                   
            
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, is_cheng)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
        return self.model(input)
    
    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-2].act_quantizer.bitwidth_refactor(8)
        # ignore reconstruction of the first layer
        # module_list[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True
