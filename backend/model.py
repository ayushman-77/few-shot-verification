import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SiameseNetwork(nn.Module):
    """
    Siamese Network architecture using a ResNet backbone for high-resolution feature extraction.
    """
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )

    def forward_once(self, x):
        """
        Forward pass for a single image to get its embedding.
        """
        output = self.cnn(x)
        output = output.view(output.size()[0], -1) # Flatten the output
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        """
        Forward pass for both images to get their respective embeddings.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
