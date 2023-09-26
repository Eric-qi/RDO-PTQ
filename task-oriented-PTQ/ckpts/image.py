import sys
import torch
sys.path.append("..")
from utils import get_config
from models.nic_cvt import NIC
from .pretrained import load_pretrained

__all__ = [
    "nic_tic",
]

model_architectures = {
    "nic": NIC,
}

models = {
    "nic": NIC,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_url = "./ckpts/Lu2022"
model_urls = {
    "nic": {
        "mse": {
            1: f"{root_url}/mse/nic_mse_1.pt",
            2: f"{root_url}/mse/nic_mse_2.pt",
            3: f"{root_url}/mse/nic_mse_3.pt",
            4: f"{root_url}/mse/nic_mse_4.pt",
            5: f"{root_url}/mse/nic_mse_5.pt",
            6: f"{root_url}/mse/nic_mse_6.pt",
            7: f"{root_url}/mse/nic_mse_7.pt",
            8: f"{root_url}/mse/nic_mse_8.pt",
        },
        "ms-ssim": {
            1: f"{root_url}/ms-ssim/nic_ms-ssim_1.pt",
            2: f"{root_url}/ms-ssim/nic_ms-ssim_2.pt",
            3: f"{root_url}/ms-ssim/nic_ms-ssim_3.pt",
            4: f"{root_url}/ms-ssim/nic_ms-ssim_4.pt",
            5: f"{root_url}/ms-ssim/nic_ms-ssim_5.pt",
            6: f"{root_url}/ms-ssim/nic_ms-ssim_6.pt",
            7: f"{root_url}/ms-ssim/nic_ms-ssim_7.pt",
            8: f"{root_url}/ms-ssim/nic_ms-ssim_8.pt",
        },
    },
}

cfgs = {
    "nic": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (192, 320),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
}

def _load_model(
    architecture, metric, quality, pretrained=False, progress=True, **kwargs
):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        if (
            architecture not in model_urls
            or metric not in model_urls[architecture]
            or quality not in model_urls[architecture][metric]
        ):
            raise RuntimeError("Pre-trained model not yet available")

        url = model_urls[architecture][metric][quality]
        print("Loading Ckpts From:", url)
        # state_dict = load_state_dict_from_url(url, progress=progress)
        state_dict = torch.load(url, map_location=torch.device(device))
        # state_dict = load_pretrained(state_dict)

        config = get_config("config.yaml")
        config['embed_dim'] = cfgs[architecture][quality][0]
        config['latent_dim'] = cfgs[architecture][quality][1]
        model = model_architectures[architecture](config)
        model.load_state_dict(state_dict['model'])
        
        # TODO: should be put in traning loop
        model.update()
        
        # model = model_architectures[architecture].from_state_dict(state_dict)
        return model

    # model = model_architectures[architecture](*cfgs[architecture][quality], **kwargs)
    # return model

def nic_tic(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""
        Neural image compression framework from Lu, Ming and Guo, Peiyao and Shi, Huiqing and Cao, Chuntong and Ma, Zhan: 
        `"Transformer-based Image Compression" <https://arxiv.org/abs/2111.06707>`, (DCC 2022).
        `"High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation" <https://arxiv.org/abs/2204.11448>`.

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
    """
    if metric not in ("mse","ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("nic", metric, quality, pretrained, progress, **kwargs)