import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers.layers import ResidualBlockWithStride, ResidualBlockUpsample, ResidualBlock, subpel_conv3x3

# sys.path.append(".")
from quantization.quant_layer import QuantModule
from quantization.quantizer import StraightThrough, UniformAffineQuantizer, ActQuantizer


from models.nic_cvt import NIC
from models.layers import Mlp, WindowAttention, SwinTransformerBlock, BasicLayer, RSTB

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform some activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.trained = False

        self.act_quantizer = UniformAffineQuantizer(act=True, **act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantNIC(BaseQuantBlock):
    """
    Implementation of Quantized NIC Lu2022.
    TODO: update
    """
    def __init__(self, basic_block: NIC, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)

        self.g_a0 = QuantModule(basic_block.g_a0, weight_quant_params, act_quant_params, disable_act_quant=False)
        self.g_a1 = basic_block.g_a1
        self.g_a2 = QuantModule(basic_block.g_a2, weight_quant_params, act_quant_params, disable_act_quant=False)
        self.g_a3 = basic_block.g_a3
        self.g_a4 = QuantModule(basic_block.g_a4, weight_quant_params, act_quant_params, disable_act_quant=False)
        self.g_a5 = basic_block.g_a5
        self.g_a6 = QuantModule(basic_block.g_a6, weight_quant_params, act_quant_params, disable_act_quant=False)
        self.g_a7 = basic_block.g_a7 # disable_act_quant of last layer can be True

        self.h_a0 = QuantModule(basic_block.h_a0, weight_quant_params, act_quant_params, disable_act_quant=False)
        self.h_a1 = basic_block.h_a1
        self.h_a2 = QuantModule(basic_block.h_a2, weight_quant_params, act_quant_params, disable_act_quant=False)
        self.h_a3 = basic_block.h_a3

        self.h_s0 = basic_block.h_s0
        self.h_s1 = QuantModule(basic_block.h_s1, weight_quant_params, act_quant_params, disable_act_quant=False)
        self.h_s2 = basic_block.h_s2
        self.h_s3 = QuantModule(basic_block.h_s3, weight_quant_params, act_quant_params, disable_act_quant=False)

        self.g_s0 = basic_block.g_s0
        self.g_s1 = QuantModule(basic_block.g_s1, weight_quant_params, act_quant_params, disable_act_quant=False)
        self.g_s2 = basic_block.g_s2
        self.g_s3 = QuantModule(basic_block.g_s3, weight_quant_params, act_quant_params, disable_act_quant=False)
        self.g_s4 = basic_block.g_s4
        self.g_s5 = QuantModule(basic_block.g_s5, weight_quant_params, act_quant_params, disable_act_quant=False)
        self.g_s6 = basic_block.g_s6
        self.g_s7 = QuantModule(basic_block.g_s7, weight_quant_params, act_quant_params, disable_act_quant=True)

        self.entropy_bottleneck = basic_block.entropy_bottleneck
        self.gaussian_conditional = basic_block.gaussian_conditional
        # mask_conv2d will be update later
        # self.context_prediction = QuantBasicBlock(basic_block.context_prediction, weight_quant_params, 
                                                  # act_quant_params, disable_act_quant=False)
        self.context_prediction = basic_block.context_prediction

        self.entropy_parameters = basic_block.entropy_parameters # TODO


    def g_a(self, x, x_size=None):
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)
        x = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        x = self.g_a2(x)
        x = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        x = self.g_a4(x)
        x = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        x = self.g_a6(x)
        x = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        return x
    
    def g_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        x = self.g_s1(x)
        x = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        x = self.g_s3(x)
        x = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        x = self.g_s5(x)
        x = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        x = self.g_s7(x)
        return x
    
    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x
    
    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x, x_size)
        z = self.h_a(y, x_size)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat, x_size)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, x_size)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


class QuantRBWS(BaseQuantBlock):
    """
    Implementation of Quantized ResidualBlockWithStride used in Cheng2020.
    """
    def __init__(self, basic_block: ResidualBlockWithStride, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.leaky_relu = basic_block.leaky_relu
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params)
        self.gdn = QuantModule(basic_block.gdn, weight_quant_params, act_quant_params)
        # self.gdn = basic_block.gdn
        if basic_block.skip is not None:
            self.skip = QuantModule(basic_block.skip, weight_quant_params, act_quant_params)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        if self.use_act_quant and self.trained:
            out = ActQuantizer(out)
        out = self.conv2(out)
        out = self.gdn(out)
        if self.skip is not None:
            identity = self.skip(x)
        out += identity
        if self.use_act_quant and self.trained:
            out = ActQuantizer(out)
        return out 
    

class QuantRBU(BaseQuantBlock):
    """
    Implementation of Quantized ResidualBlockUpsample used in Cheng2020.
    """
    def __init__(self, basic_block: ResidualBlockUpsample, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.subpel_conv = nn.Sequential(
            QuantModule(basic_block.subpel_conv[0], weight_quant_params, act_quant_params, disable_act_quant=True),
            basic_block.subpel_conv[1]
            )
        self.leaky_relu = basic_block.leaky_relu
        self.conv = QuantModule(basic_block.conv, weight_quant_params, act_quant_params)
        self.igdn = QuantModule(basic_block.igdn, weight_quant_params, act_quant_params)
        # self.igdn = basic_block.igdn
        self.upsample = nn.Sequential(
            QuantModule(basic_block.upsample[0], weight_quant_params, act_quant_params),
            basic_block.upsample[1]
            )

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        if self.use_act_quant and self.trained:
            out = ActQuantizer(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        if self.use_act_quant and self.trained:
            out = ActQuantizer(out)
        return out 

class QuantRB(BaseQuantBlock):
    """
    Implementation of Quantized ResidualBlock used in Cheng2020.
    """
    def __init__(self, basic_block: ResidualBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.leaky_relu = basic_block.leaky_relu
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)
        if basic_block.skip is not None:
            self.skip = QuantModule(basic_block.skip, weight_quant_params, act_quant_params)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        if self.use_act_quant and self.trained:
            out = ActQuantizer(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        if self.use_act_quant and self.trained:
            out = ActQuantizer(out)
        if self.skip is not None:
            identity = self.skip(x)
        out = out + identity
        if self.use_act_quant and self.trained:
            out = ActQuantizer(out)
        return out     

class QuantSC(BaseQuantBlock):
    """
    Implementation of Quantized subpel_conv3x3 used in Cheng2020.
    """
    def __init__(self, basic_block: subpel_conv3x3, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.subpel_conv = nn.Sequential(
            QuantModule(basic_block[0], weight_quant_params, act_quant_params, disable_act_quant=True),
            basic_block[1],
            nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):
        return self.subpel_conv(x)   

class QuantMlp(BaseQuantBlock):
    """
    Implementation of Quantized Mlp used in Lu2022.
    """
    def __init__(self, basic_block: Mlp, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.fc1 = QuantModule(basic_block.fc1, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.act = basic_block.act
        self.fc2 = QuantModule(basic_block.fc2, weight_quant_params, act_quant_params)
        # self.activation_function = basic_block.drop

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.use_act_quant and self.trained:
            x = ActQuantizer(x)
        x = self.fc2(x)
        return x


class QuantWindowAttention(BaseQuantBlock):
    """
    Implementation of Quantized WindowAttention used in Lu2022.
    """
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, basic_block: WindowAttention, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)

        self.dim = basic_block.dim
        self.window_size = basic_block.window_size  # Wh, Ww
        self.num_heads = basic_block.num_heads
        head_dim = basic_block.dim // basic_block.num_heads
        self.scale = basic_block.scale

        self.qkv = QuantModule(basic_block.qkv, weight_quant_params, act_quant_params)
        self.attn_drop = basic_block.attn_drop
        self.proj = QuantModule(basic_block.proj, weight_quant_params, act_quant_params)
        self.proj_drop = basic_block.proj_drop
        self.softmax = basic_block.softmax

        self.relative_position_bias_table = basic_block.relative_position_bias_table
        self.relative_position_index = basic_block.relative_position_index # whether True??

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        if self.use_act_quant and self.trained:
            attn = ActQuantizer(attn)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.use_act_quant and self.trained:
            x = ActQuantizer(x)

        x = self.proj(x)
        # x = self.proj_drop(x)

        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class QuantSwinTransformerBlock(BaseQuantBlock):
    """
    Implementation of Quantized Swin Transformer Block used in Lu2022.
    """
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, basic_block: SwinTransformerBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.dim = basic_block.dim
        self.input_resolution = basic_block.input_resolution
        self.num_heads = basic_block.num_heads
        self.window_size = basic_block.window_size
        self.shift_size = basic_block.shift_size
        self.mlp_ratio = basic_block.mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = QuantModule(basic_block.norm1, weight_quant_params, act_quant_params)
        self.attn = QuantWindowAttention(basic_block.attn, weight_quant_params, act_quant_params)
        #self.attn = basic_block.attn
        # self.drop_path = basic_block.drop_path
        # self.drop_path = nn.Identity()
        self.norm2 = QuantModule(basic_block.norm2, weight_quant_params, act_quant_params)
        self.mlp = QuantMlp(basic_block.mlp, weight_quant_params, act_quant_params)
        #self.mlp = basic_block.mlp

        # if self.shift_size > 0:
        #     attn_mask = self.calculate_mask(self.input_resolution)
        # else:
        #     attn_mask = None

        self.attn_mask = basic_block.attn_mask

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        # x = shortcut + self.drop_path(x)
        x = shortcut + x
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.mlp(self.norm2(x))
        # print("STB",x.shape, x.dtype)

        if self.use_act_quant and self.trained:
            x = ActQuantizer(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class QuantBasicLayer(BaseQuantBlock):
    """
    Implementation of Quantized BasicLayer used in Lu2022.
    """
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, basic_block: BasicLayer, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.dim = basic_block.dim
        self.input_resolution = basic_block.input_resolution
        self.depth = basic_block.depth
        self.use_checkpoint = basic_block.use_checkpoint

        self.blocks = nn.ModuleList([
            QuantSwinTransformerBlock(basic_block.blocks[i], weight_quant_params, act_quant_params)
            for i in range(self.depth)])
        # self.blocks = nn.ModuleList([basic_block.blocks[i] for i in range(self.depth)])
                                        

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class QuantRSTB(BaseQuantBlock):
    """
    Implementation of Quantized RSTB used in Lu2022.
    """
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, basic_block: RSTB, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.dim = basic_block.dim
        self.input_resolution = basic_block.input_resolution
       
        self.residual_group = QuantBasicLayer(basic_block.residual_group, weight_quant_params, act_quant_params)
        # self.residual_group = basic_block.residual_group

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()

    def forward(self, x, x_size):
        out = self.patch_unembed(self.residual_group(self.patch_embed(x), x_size), x_size) + x

        if self.use_act_quant and self.trained:
            out = ActQuantizer(out)
        return out



specials = {
    #NIC: QuantNIC,
    #Mlp: QuantMlp,
    #WindowAttention: QuantWindowAttention,
    #SwinTransformerBlock: QuantSwinTransformerBlock,
    #BasicLayer: QuantBasicLayer,
    RSTB: QuantRSTB,
    
    ResidualBlockWithStride: QuantRBWS,
    ResidualBlockUpsample: QuantRBU,
    ResidualBlock: QuantRB,
    subpel_conv3x3: QuantSC,
}
