# src/models/gating.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

class GatingFeatureBuilder:
    """
    Builds scalable, class-count-independent features from expert posteriors/logits.
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @torch.no_grad()
    def __call__(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_logits: Tensor of shape [B, E, C] (Batch, Experts, Classes)
        
        Returns:
            A feature tensor of shape [B, D] where D is the feature dimension.
        """
        # Ensure input is float32 for stable calculations
        expert_logits = expert_logits.float()
        B, E, C = expert_logits.shape
        
        # Use posteriors for probability-based features
        expert_posteriors = torch.softmax(expert_logits, dim=-1)

        # Feature 1: Entropy of each expert's prediction
        # Shape: [B, E]
        entropy = -torch.sum(expert_posteriors * torch.log(expert_posteriors + 1e-8), dim=-1)

        # Feature 2: Top-k probability mass and residual mass
        # Shape: [B, E, K], we take top-k values
        topk_vals, _ = torch.topk(expert_posteriors, k=self.top_k, dim=-1)
        # Shape: [B, E]
        topk_mass = torch.sum(topk_vals, dim=-1)
        # Shape: [B, E]
        residual_mass = 1.0 - topk_mass

        # Feature 3: Pairwise cosine similarity between expert posteriors
        # A simplified way to measure agreement without O(E^2) complexity
        # is to compare each expert to the ensemble mean.
        # Shape: [B, C]
        mean_posterior = torch.mean(expert_posteriors, dim=1)
        # Shape: [B, E]
        cosine_sim = F.cosine_similarity(expert_posteriors, mean_posterior.unsqueeze(1), dim=-1)

        # Concatenate all features
        # [B, E], [B, E], [B, E], [B, E] -> [B, 4*E]
        features = torch.cat([entropy, topk_mass, residual_mass, cosine_sim], dim=1)
        
        return features

class GatingNet(nn.Module):
    """
    A simple MLP that takes gating features and outputs expert weights.
    """
    def __init__(self, in_dim: int, hidden_dims: list = [128, 64], num_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Gating features of shape [B, D]
        
        Returns:
            Expert weights (before softmax) of shape [B, E]
        """
        return self.net(x)