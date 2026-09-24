import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function for Siamese Network training.
    Encourages similar items to have a small distance, and dissimilar items to have a distance greater than a margin.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Computes the contrastive loss.
        label == 0: Similar images (same class)
        label == 1: Dissimilar images (different classes)
        """
        # Remove keepdim=True so euclidean_distance has shape [batch_size]
        # This matches the shape of label [batch_size], preventing a [batch_size, batch_size] broadcasting bug!
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
