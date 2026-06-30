"""Siamese embedding network (GraphBased paper Step 3a).

ResNet-50 (ImageNet-pretrained) with the backbone FROZEN, topped with two dense
layers of `dim` neurons (ReLU), producing a d-dimensional embedding. Trained with
triplet loss so the embedding is Euclidean-metric-calibrated (needed for the graph
edge threshold tau=0.5 in Step 3b).

Paper: "each twin ... ResNet-50 ... freeze the weights of all pre-trained layers ...
add our own 2 dense layers with d neurons ... Euclidean distance ... triplet loss."
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SiameseEmbedding(nn.Module):
    def __init__(self, dim=2048, freeze_backbone=True, normalize=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feat_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # two dense layers of `dim` neurons (paper Step 3a)
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.normalize = normalize

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                f = self.backbone(x)
        else:
            f = self.backbone(x)
        e = self.head(f)
        if self.normalize:
            e = F.normalize(e, p=2, dim=1)
        return e

    def trainable_parameters(self):
        return self.head.parameters()
