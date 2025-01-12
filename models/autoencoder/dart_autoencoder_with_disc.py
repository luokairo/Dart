from turtle import forward
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from dart.losses.vqgan import create_discriminator

from dart.models.autoencoder.dart_configuration import DARTAutoEncoderWithDiscConfig
from dart.models.autoencoder.dart_autoencoder import DARTAutoEncoder


class DARTAutoEncoderWithDisc(PreTrainedModel):
    config_class = DARTAutoEncoderWithDiscConfig

    def __init__(
        self, 
        config: DARTAutoEncoderWithDiscConfig
    ):
        super().__init__(config)

        self.vae = DARTAutoEncoder(config) 
        self.disc = create_discriminator()

    def forward(
        self,
        *,
        model_index=0,
        **kwargs,
    ):
        if model_index == 0:
            return self.vae(**kwargs)
        else:
            return self.disc(**kwargs)
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight

    def compute_loss(self, out, usages, vq_loss, recon_weight=1.0, xs=None):
        loss_recon = F.mse_loss(out, xs, reduction="mean")

        loss_latent = vq_loss

        loss_total = loss_recon * recon_weight + loss_latent

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_latent": loss_latent,
        }

    @property
    def vae_params(self):
        return self.vae.parameters()

    @property
    def disc_params(self):
        return self.disc.parameters()
    
AutoConfig.register("dart_autoencoder_with_disc", DARTAutoEncoderWithDisc)
AutoModel.register(DARTAutoEncoderWithDiscConfig, DARTAutoEncoderWithDisc)