from ast import Tuple
import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from dart.models.transformer.dart_transformer import DARTForT2I
from dart.models.autoencoder import (
    DARTAutoEncoder,
    DartHybridQuantizer,
    DARTAutoEncoderWithDisc,
)
from dart.models.router import (
    DARTRouterMlp,
)

class DARTTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: DARTAutoEncoderWithDisc, var_wo_dpp: VAR, var: DDP,
    ):
        