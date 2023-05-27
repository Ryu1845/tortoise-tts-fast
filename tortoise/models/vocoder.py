import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch.nn as nn

try:
    from BigVGAN.env import AttrDict
    from BigVGAN.models import BigVGAN as BVGModel
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
