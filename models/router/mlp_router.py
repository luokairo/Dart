import logging
from typing import Literal

import einops
import torch
import torch.nn as nn
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrConfig, DeformableDetrDecoderLayer)

class DARTRouterMlp(nn.Module):
    def __init__(
        self,
        seq_dim=512,
        embedding=1000,
        num_selector=2,
    ):
        super().__init__()
        self.num_selector = num_selector
        self.routing_outcome = None

        self.router = nn.Sequential(
            nn.Linear(embedding, seq_dim),
            nn.GELU(),
            nn.Linear(seq_dim, seq_dim),
            nn.GELU(),
            nn.Linear(seq_dim, self.num_selector),
        )
    
    def forward(self, con_embeds, dis_embeds, seq_embeds):
        """
        Args:
            con_embeds: Continuous embeddings (batch_size, seq_len, embed_dim)
            dis_embeds: Discrete embeddings (batch_size, seq_len, embed_dim)
            seq_embeds: Routing input embeddings (batch_size, context_tokens, seq_dim)
        """
        hidden_states = torch.zeros_like(con_embeds)
        batch_size = seq_embeds.shape[0]

        ratios = self.router(seq_embeds.view(batch_size, -1)) 
        ratios = torch.softmax(ratios, dim=-1)

        con_weighted_embeds = ratios[:, 0].view(-1, 1, 1) * con_embeds
        dis_weighted_embeds = ratios[:, 1].view(-1, 1, 1) * dis_embeds

        hidden_states += con_weighted_embeds
        hidden_states += dis_weighted_embeds

        if self.routing_outcome is None:
            self.routing_outcome = {}
        self.routing_outcome["continuous"] = con_weighted_embeds.abs().mean(dim=-1).mean(dim=-1) 
        self.routing_outcome["discrete"] = dis_weighted_embeds.abs().mean(dim=-1).mean(dim=-1)

        return hidden_states