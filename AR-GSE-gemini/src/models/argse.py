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
        self.alpha = nn.Parameter(torch.full((num_groups,), 1.0))  # Start with reasonable confidence scaling
        # Initialize mu to negative values so threshold = c + mu is less than c (easier to accept)
        self.mu = nn.Parameter(torch.full((num_groups,), -0.5))
        
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

        # 2. Apply temperature scaling to expert logits before mixture
        # Temperature scaling helps with calibration (typical temperature ~2-4)
        temperature = 2.0  # Reduce temperature for less conservative predictions
        expert_logits_scaled = expert_logits / temperature
        
        # 3. Mixture with temperature-scaled logits
        expert_posteriors = F.softmax(expert_logits_scaled, dim=-1)
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

        # Score: max_y alpha_g(y) * eta_y (NOT divided by alpha)
        # This encourages acceptance when confidence is high AND alpha is high
        score_per_class = alpha[class_to_group] * eta_mix  # [B, C]
        max_score, _ = score_per_class.max(dim=1)  # [B]
        
        # Threshold: c + mu_g (simplified threshold)
        # Get the predicted class for threshold calculation
        _, pred_class = eta_mix.max(dim=1)  # [B]
        pred_groups = class_to_group[pred_class]  # [B]
        threshold = c + mu[pred_groups]  # [B]
        
        # Margin: score - threshold
        # Positive margin means accept, negative means reject
        margin = max_score - threshold
        return margin