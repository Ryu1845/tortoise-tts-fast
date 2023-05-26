import torch
import torch.nn as nn

import json
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass

try:
    from BigVGAN.models import BigVGAN as BVGModel
    from BigVGAN.env import AttrDict
except ImportError:
    raise ImportError(
        "BigVGAN not installed, can't use BigVGAN vocoder\n"
        "Please see the installation instructions on README."
    )

MAX_WAV_VALUE = 32768.0


from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent.parent / "static"
assert STATIC_DIR.is_dir()


def BVGWithConf(fname: str):
    json_config = json.loads((STATIC_DIR / fname).read_text())
    return lambda: BVGModel(AttrDict(json_config))


@dataclass
class VocType:
    constructor: Callable[[], nn.Module]
    model_path: str
    subkey: Optional[str] = None

    def optionally_index(self, model_dict):
        if self.subkey is not None:
            return model_dict[self.subkey]
        return model_dict


class VocConf(Enum):
    BigVGAN_Base = VocType(
        BVGWithConf("bigvgan_base_24khz_100band_config.json"),
        "bigvgan_base_24khz_100band_g.pth",
        "generator",
    )
    BigVGAN = VocType(
        BVGWithConf("bigvgan_24khz_100band_config.json"),
        "bigvgan_24khz_100band_g.pth",
        "generator",
    )


if __name__ == "__main__":
    model = UnivNetGenerator()

    c = torch.randn(3, 100, 10)
    z = torch.randn(3, 64, 10)
    print(c.shape)

    y = model(c, z)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
