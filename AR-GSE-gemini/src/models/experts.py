# src/models/experts.py
import torch
import torch.nn as nn
from .backbones.resnet_cifar import ResNet32

class Expert(nn.Module):
    def __init__(self, num_classes=100, backbone_name='resnet32'):
        super(Expert, self).__init__()
        if backbone_name == 'resnet32':
            self.backbone = ResNet32()
            # The output feature dimension of our ResNet32 is 64
            self.fc = nn.Linear(64, num_classes)
        else:
            raise ValueError(f"Backbone '{backbone_name}' not recognized.")
            
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits
    
    def get_calibrated_logits(self, x):
        return self.forward(x) / self.temperature

    def set_temperature(self, temp):
        self.temperature.data = torch.tensor(temp)