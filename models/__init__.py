from dart.models.transformer.configuration import DARTForT2IConfig
from dart.models.autoencoder.dart_configuration import DARTAutoEncoderWithDiscConfig
from dart.models.transformer.dart_transformer import DARTForT2I
from dart.models.autoencoder.dart_autoencoder_with_disc import DARTAutoEncoderWithDisc, DARTAutoEncoder

from typing import Tuple
import torch.nn as nn

def build_vae_dart(
    # Shared args
    device,
    # VQVAE args
    dart_autoencoder_config=DARTAutoEncoderWithDiscConfig,
    # Dart args
    dart_config=DARTForT2IConfig,
    
) -> Tuple[DARTAutoEncoderWithDisc, DARTForT2I]:
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)

    vae_loacl = DARTAutoEncoderWithDisc(dart_autoencoder_config).vae.to(device)
    dart_wo_ddp = DARTForT2I(dart_config).to(device)

    return vae_loacl, dart_wo_ddp