import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure


class SSIMLoss(nn.Module):
    # SSIM loss between two images
    def __init__(self, data_range=1.0):
        super(SSIMLoss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
        
    def forward(self, x, y):
        return 1 - self.ssim(x, y)
        

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
