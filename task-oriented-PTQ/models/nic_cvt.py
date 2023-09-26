import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import MaskedConv2d
from timm.models.layers import trunc_normal_
from .layers import RSTB
from .utils import update_registered_buffers

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class NIC(nn.Module):
    def __init__(self, config):
        super().__init__()

        height = config['height']
        width = config['width']
        in_chans = config['in_chans']
        embed_dim = config['embed_dim']
        latent_dim = config['latent_dim']
        window_size = config['window_size']
        mlp_ratio = config['mlp_ratio']
        qkv_bias = config['qkv_bias']
        qk_scale = config['qk_scale']
        drop_rate = config['drop_rate']
        attn_drop_rate = config['attn_drop_rate']
        drop_path_rate = config['drop_path_rate']
        use_checkpoint = config['use_checkpoint']
        norm_layer = nn.LayerNorm
        
        self.M = latent_dim

        depths = [2, 4, 6, 2, 2, 2, 2, 2, 2, 6, 4, 2]
        num_heads = [4, 8, 8, 16, 16, 16, 16, 16, 16, 8, 8, 4]

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:len(depths)//2]))] 
        dec_dpr = enc_dpr[::-1]

        self.g_a0 = nn.Conv2d(in_chans, embed_dim, kernel_size=5, stride=2, padding=2)
        self.g_a1 = RSTB(dim=embed_dim,
                        input_resolution=(height//2,
                                        width//2),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_a2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.g_a3 = RSTB(dim=embed_dim,
                        input_resolution=(height//4,
                                        width//4),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_a4 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.g_a5 = RSTB(dim=embed_dim,
                        input_resolution=(height//8,
                                        width//8),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_a6 = nn.Conv2d(embed_dim, latent_dim, kernel_size=3, stride=2, padding=1)
        self.g_a7 = RSTB(dim=latent_dim,
                        input_resolution=(height//16,
                                        width//16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )

        self.h_a0 = nn.Conv2d(latent_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.h_a1 = RSTB(dim=embed_dim,
                         input_resolution=(height//32,
                                        width//32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=enc_dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_a2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.h_a3 = RSTB(dim=embed_dim,
                         input_resolution=(height//64,
                                        width//64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=enc_dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )

        self.h_s0 = RSTB(dim=embed_dim,
                         input_resolution=(height//64,
                                        width//64),
                         depth=depths[6],
                         num_heads=num_heads[6],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dec_dpr[:depths[6]],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_s1 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.h_s2 = RSTB(dim=embed_dim,
                         input_resolution=(height//32,
                                        width//32),
                         depth=depths[7],
                         num_heads=num_heads[7],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dec_dpr[sum(depths[6:7]):sum(depths[6:8])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_s3 = nn.ConvTranspose2d(embed_dim, latent_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.g_s0 = RSTB(dim=latent_dim,
                        input_resolution=(height//16,
                                        width//16),
                        depth=depths[8],
                        num_heads=num_heads[8],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dec_dpr[sum(depths[6:8]):sum(depths[6:9])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s1 = nn.ConvTranspose2d(latent_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.g_s2 = RSTB(dim=embed_dim,
                        input_resolution=(height//8,
                                        width//8),
                        depth=depths[9],
                        num_heads=num_heads[9],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dec_dpr[sum(depths[6:9]):sum(depths[6:10])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s3 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.g_s4 = RSTB(dim=embed_dim,
                        input_resolution=(height//4,
                                        width//4),
                        depth=depths[10],
                        num_heads=num_heads[10],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dec_dpr[sum(depths[6:10]):sum(depths[6:11])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s5 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.g_s6 = RSTB(dim=embed_dim,
                        input_resolution=(height//2,
                                        width//2),
                        depth=depths[11],
                        num_heads=num_heads[11],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dec_dpr[sum(depths[6:11]):sum(depths[6:12])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s7 = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=5, stride=2, padding=2, output_padding=1)
        
        self.entropy_bottleneck = EntropyBottleneck(embed_dim)
        self.gaussian_conditional = GaussianConditional(None)    
        self.context_prediction = MaskedConv2d(latent_dim, latent_dim*2, kernel_size=5, padding=2, stride=1)   

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(latent_dim*12//3, latent_dim*10//3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_dim*10//3, latent_dim*8//3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_dim*8//3, latent_dim*6//3, 1),
        )   

        self.apply(self._init_weights)   

    # g_a, encoder 
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

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

    def flops(self, height, width, in_chans, embed_dim, latent_dim):
        flops = 0

        g_a_flops = 0
        g_a_flops += self.g_a1.flops()
        g_a_flops += self.g_a3.flops()
        g_a_flops += self.g_a5.flops()
        g_a_flops += self.g_a7.flops()
        g_a_flops += height//2*width//2*embed_dim*in_chans*5*5
        g_a_flops += height//4*width//4*embed_dim*embed_dim*3*3
        g_a_flops += height//8*width//8*embed_dim*embed_dim*3*3
        g_a_flops += height//16*width//16*latent_dim*embed_dim*3*3

        g_s_flops = 0
        g_s_flops += self.g_s0.flops()
        g_s_flops += self.g_s2.flops()
        g_s_flops += self.g_s4.flops()
        g_s_flops += self.g_s6.flops()
        g_s_flops += height//8*width//8*embed_dim*latent_dim*3*3
        g_s_flops += height//4*width//4*embed_dim*embed_dim*3*3
        g_s_flops += height//2*width//2*embed_dim*embed_dim*3*3
        g_s_flops += height*width*in_chans*embed_dim*5*5

        h_a_flops = 0
        h_a_flops += self.h_a1.flops()
        h_a_flops += self.h_a3.flops()
        h_a_flops += height//32*width//32*embed_dim*latent_dim*3*3
        h_a_flops += height//64*width//64*embed_dim*embed_dim*3*3

        h_s_flops = 0
        h_s_flops += self.h_s0.flops()
        h_s_flops += self.h_s2.flops()
        h_s_flops += height//32*width//32*embed_dim*embed_dim*3*3
        h_s_flops += height//16*width//16*latent_dim*2*embed_dim*3*3

        flops = g_a_flops + g_s_flops + h_a_flops + h_s_flops

        return g_a_flops/1e9, g_s_flops/1e9, h_a_flops/1e9, h_s_flops/1e9, flops/1e9

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def parameters(self):
        """Returns an iterator over the model parameters."""
        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def aux_parameters(self):
        """
        Returns an iterator over the entropy bottleneck(s) parameters for
        the auxiliary loss.
        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
                check if we need to update the gaussian conditional parameters, 
                the offsets are only computed and stored when the conditonal model is updated.
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()
        # pylint: enable=protected-access
        # print(cdf, cdf_lengths, offsets)
        y_strings = []
        for i in range(y.size(0)):
            encoder = BufferedRansEncoder()
            # Warning, this is slow...
            # TODO: profile the calls to the bindings...
            symbols_list = []
            indexes_list = []
            for h in range(y_height):
                for w in range(y_width):
                    y_crop = y_hat[
                        i : i + 1, :, h : h + kernel_size, w : w + kernel_size
                    ]
                    ctx_p = F.conv2d(
                        y_crop,
                        self.context_prediction.weight,
                        bias=self.context_prediction.bias,
                    )

                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i : i + 1, :, h : h + 1, w : w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)
                    y_q = torch.round(y_crop - means_hat)
                    y_hat[i, :, h + padding, w + padding] = (y_q + means_hat)[
                        i, :, padding, padding
                    ]

                    symbols_list.extend(y_q[i, :, padding, padding].int().tolist())
                    indexes_list.extend(indexes[i, :].squeeze().int().tolist())

            encoder.encode_with_indexes(
                symbols_list, indexes_list, cdf, cdf_lengths, offsets
            )

            string = encoder.flush()
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        decoder = RansDecoder()

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for i, y_string in enumerate(strings[0]):
            decoder.set_stream(y_string)

            for h in range(y_height):
                for w in range(y_width):
                    # only perform the 5x5 convolution on a cropped tensor
                    # centered in (h, w)
                    y_crop = y_hat[
                        i : i + 1, :, h : h + kernel_size, w : w + kernel_size
                    ]
                    ctx_p = F.conv2d(
                        y_crop,
                        self.context_prediction.weight,
                        bias=self.context_prediction.bias,
                    )
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i : i + 1, :, h : h + 1, w : w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)

                    rv = decoder.decode_stream(
                        indexes[i, :].squeeze().int().tolist(),
                        cdf,
                        cdf_lengths,
                        offsets,
                    )
                    rv = torch.Tensor(rv).reshape(1, -1, 1, 1)

                    rv = self.gaussian_conditional._dequantize(rv, means_hat)

                    y_hat[
                        i,
                        :,
                        h + padding : h + padding + 1,
                        w + padding : w + padding + 1,
                    ] = rv
        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        # pylint: enable=protected-access

        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
        
