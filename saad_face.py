import torch.nn as nn
from torchvision import models

class SaadFace(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SaadFace, self).__init__()
        model = models.mobilenet_v3_small(pretrained=True)  # Use ResNet-18 as a base
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 128)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return nn.functional.normalize(x, p=2, dim=1)  # Normalize embeddings
