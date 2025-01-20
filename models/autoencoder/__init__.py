from .dart_autoencoder import *
from .configuration import *
from .dart_autoencoder_with_disc import *
from dart.models.transformer.configuration import DARTForT2IConfig
from dart.models.autoencoder.dart_configuration import DARTAutoEncoderConfig

from typing import Tuple
import torch.nn as nn

def build_vae_dart(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    # VQVAE args
    dart_config=DARTForT2IConfig,
    
) -> 