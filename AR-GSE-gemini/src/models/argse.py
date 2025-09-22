# src/models/argse.py
import torch
import torch.nn as nn
from .gating import GatingFeatureBuilder, GatingNet
import torch.nn.functional as F

class AR_GSE(nn.Module):
    def __init__(self, num_experts: int, num_classes: int, num_groups: int, gating_feature_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.num_groups = num_groups

        # Gating components
        self.feature_builder = GatingFeatureBuilder()
        self.gating_net = GatingNet(in_dim=gating_feature_dim, num_experts=num_experts)

        # Primal variables (learnable parameters)
        # Initialize alpha to 1 for all groups
        self.alpha = nn.Parameter(torch.ones(num_groups))
        # Initialize mu to 0 for all groups
        self.mu = nn.Parameter(torch.zeros(num_groups))
        
        # Dual variables (not optimized by SGD, but part of state)
        self.register_buffer('Lambda', torch.zeros(num_groups))

    def forward(self, expert_logits, c, tau, class_to_group):
        """
        Full forward pass to get all necessary components for the primal-dual loss.
        
        Args:
            expert_logits (Tensor): Shape [B, E, C]
            c (float): Rejection cost
            tau (float): Temperature for sigmoid
            class_to_group (LongTensor): Shape [C]

        Returns:
            A dictionary of tensors for loss computation.
        """
        # 1. Gating
        gating_features = self.feature_builder(expert_logits)
        gating_raw_weights = self.gating_net(gating_features)
        w = F.softmax(gating_raw_weights, dim=1) # Shape [B, E]

        # 2. Mixture
        expert_posteriors = F.softmax(expert_logits, dim=-1)
        # einops is great for this: b e c -> b c
        eta_mix = torch.einsum('be,bec->bc', w, expert_posteriors)
        eta_mix = torch.clamp(eta_mix, min=1e-8) # for stability

        # 3. Margin & Acceptance Probability
        margin = self.selective_margin(eta_mix, c, class_to_group)
        s_tau = torch.sigmoid(tau * margin)

        return {
            'eta_mix': eta_mix,
            's_tau': s_tau,
            'w': w,
            'margin': margin,
        }

    def selective_margin(self, eta_mix, c, class_to_group):
        """Calculates the selective margin m(x)."""
        device = eta_mix.device
        alpha = self.alpha.to(device)
        mu = self.mu.to(device)
        class_to_group = class_to_group.to(device)

        inv_a = 1.0 / (alpha + 1e-8) # [K]
        g = class_to_group # [C]
        
        # Score: max_y eta_y / alpha_g(y)
        score_per_class = eta_mix / inv_a[g] # Broadcasting [B, C] / [C] -> [B, C]
        max_score, _ = score_per_class.max(dim=1) # [B]
        
        # Threshold: sum_y' (1/alpha_g(y') - mu_g(y')) * eta_y' - c
        coeff = inv_a[g] - mu[g] # [C]
        # einsum is safer than sum(eta_mix * coeff) for broadcasting
        thr = torch.einsum('bc,c->b', eta_mix, coeff) - c # [B]
        
        margin = max_score - thr
        return margin