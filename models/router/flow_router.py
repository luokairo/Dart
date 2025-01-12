import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFlowRouter(nn.Module):
    def __init__(self, token_dim, query_dim, hidden_dim):
        """
        Args:
            token_dim: Dimension of tokens (Cvae).
            query_dim: Dimension of the text query embedding.
            hidden_dim: Hidden dimension for intermediate layers.
        """
        super(MultiScaleFlowRouter, self).__init__()
        self.token_dim = token_dim
        self.query_dim = query_dim
        
        # Router MLP for dynamic weight computation
        self.router_mlp = nn.Sequential(
            nn.Linear(token_dim * 2 + query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)  # Output weights: w_discrete and w_continuous
        )
        
        # Flow transformation for tokens
        self.flow_transform = nn.Linear(token_dim, token_dim)  # Shared space mapping

    def forward(self, token_discrete, token_continuous, text_query):
        """
        Args:
            token_discrete: Discrete tokens (batch_size, pn, token_dim).
            token_continuous: Continuous tokens (batch_size, pn, token_dim).
            text_query: Text query embedding (batch_size, query_dim).
        
        Returns:
            mix_tokens: Fused tokens for the current step (batch_size, pn, token_dim).
            weights: Weighting values for discrete and continuous tokens (batch_size, pn, 2).
        """
        batch_size, pn, token_dim = token_discrete.shape

        # Normalize tokens
        token_discrete = self.flow_transform(token_discrete)  # (batch_size, pn, token_dim)
        token_continuous = self.flow_transform(token_continuous)  # (batch_size, pn, token_dim)

        # Broadcast text_query to match token dimensions
        text_query = text_query.unsqueeze(1).expand(-1, pn, -1)  # (batch_size, pn, query_dim)

        # Concatenate inputs for MLP
        inputs = torch.cat([token_discrete, token_continuous, text_query], dim=-1)  # (batch_size, pn, token_dim*2 + query_dim)

        # Compute weights
        weights = self.router_mlp(inputs)  # (batch_size, pn, 2)
        weights = F.softmax(weights, dim=-1)  # Normalize weights

        # Apply weights to tokens
        mix_tokens = weights[..., 0:1] * token_discrete + weights[..., 1:2] * token_continuous  # (batch_size, pn, token_dim)

        return mix_tokens, weights