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
        seq_dim=512,         # 隐藏层维度
        embedding=1536,      # 初始输入维度（seq_embeds 的每个 token）
        num_selector=2,      # 路由选择器数量
        seq_len=300,         # 输入序列长度
        pooling_type="linear", # 池化类型：mean 或 max
    ):
        super().__init__()
        self.num_selector = num_selector
        self.routing_outcome = None
        self.pooling_type = pooling_type

        # 将 seq_embeds 从 (batch_size, seq_len, embedding) 转换为 (batch_size, embedding)
        self.embed_reduce = nn.Linear(seq_len, 1)

        # Router 定义
        self.router = nn.Sequential(
            nn.Linear(embedding, seq_dim),
            nn.GELU(),
            nn.Linear(seq_dim, seq_dim),
            nn.GELU(),
            nn.Linear(seq_dim, num_selector),
        )

    def forward(self, con_h_BChw, dis_h_BChw, seq_embeds):
        """
        Args:
            con_h_BChw: Continuous embeddings (batch_size, Cvae, ph, pw)
            dis_h_BChw: Discrete embeddings (batch_size, Cvae, ph, pw)
            seq_embeds: Routing input embeddings (batch_size, seq_len, embedding)
        """
        # 初始化 hidden_states，与 con_h_BChw 的形状一致
        hidden_states = torch.zeros_like(con_h_BChw)
        batch_size, Cvae, ph, pw = con_h_BChw.shape

        # 使用池化或降维处理 seq_embeds -> (batch_size, embedding)
        if self.pooling_type == "mean":
            seq_embeds_reduced = seq_embeds.mean(dim=1)  # 平均池化
        elif self.pooling_type == "max":
            seq_embeds_reduced = seq_embeds.max(dim=1)[0]  # 最大池化
        else:
            # 可选降维方式，通过 Linear 映射序列长度维度
            seq_embeds_reduced = self.embed_reduce(seq_embeds.transpose(1, 2)).squeeze(-1)

        # 通过路由器计算路由权重
        ratios = self.router(seq_embeds_reduced)  # (batch_size, num_selector)
        ratios = torch.softmax(ratios, dim=-1)   # (batch_size, num_selector)

        # 将权重扩展到 (batch_size, Cvae, ph, pw) 匹配的形状
        con_weight = ratios[:, 0].view(-1, 1, 1, 1)  # (batch_size, 1, 1, 1)
        dis_weight = ratios[:, 1].view(-1, 1, 1, 1)  # (batch_size, 1, 1, 1)

        # 计算连续和离散权重的平均值，形状为 (batch_size,)
        con_weight_mean = con_weight.view(batch_size, -1).mean(dim=-1)  # (batch_size,)
        dis_weight_mean = dis_weight.view(batch_size, -1).mean(dim=-1)  # (batch_size,)

        # 保存路由结果（用于调试或监控）
        if self.routing_outcome is None:
            self.routing_outcome = {}
        self.routing_outcome["continuous"] = con_weight_mean  # 保存连续权重的平均值
        self.routing_outcome["discrete"] = dis_weight_mean    # 保存离散权重的平均值

        # 加权合并
        con_weighted_embeds = con_weight * con_h_BChw
        dis_weighted_embeds = dis_weight * dis_h_BChw

        hidden_states += con_weighted_embeds
        hidden_states += dis_weighted_embeds

        return hidden_states, self.routing_outcome