import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class Detector(nn.Module):
    """
    A simple, clean nn.Module wrapper for the EfficientNet-b4 model.
    The optimizer and training logic are handled by the main training script.
    
    This is used as the TEACHER model in knowledge distillation.
    For paper-config training with SAM, use model_paper.py instead.
    """
    def __init__(self):
        super(Detector, self).__init__()
        
        # Add the name for logging purposes
        self.name = 'efficientnet-b4'
        
        # Define the network architecture
        self.net = EfficientNet.from_pretrained(
            "efficientnet-b4",
            advprop=True,
            num_classes=2
        )

    def forward(self, x):
        """Defines the forward pass of the model."""
        return self.net(x)
    