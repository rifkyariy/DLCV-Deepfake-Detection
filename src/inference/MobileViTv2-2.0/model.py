import torch
from torch import nn
import os
import sys

# import src/utils 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from utils.sam import SAM 

class Detector(nn.Module):
    """
    A deepfake detector model using MobileViTv2-2.0 as the backbone.
    Can be initialized for training (with optimizer/loss) or inference.
    """

    def __init__(self, lr=0.001, num_classes=2):
        """
        Initializes the Detector model.

        Args:
            lr (float): The learning rate for the optimizer.
            num_classes (int): The number of output classes for the classifier.
        """
        super(Detector, self).__init__()
        
        # --- Load MobileViTv2-2.0 using timm ---
        try:
            import timm
            self.net = timm.create_model('mobilevitv2_200', pretrained=True, num_classes=num_classes)
        except ImportError:
            print("timm not installed. Installing timm...")
            os.system("pip install timm")
            import timm
            self.net = timm.create_model('mobilevitv2_200', pretrained=True, num_classes=num_classes)
        except Exception as e:
            print(f"Failed to load model with timm: {e}")
            print("Falling back to manual model construction...")
            from torchvision import models
            self.net = models.mobilenet_v3_large(pretrained=True)
            self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, num_classes)
        
        # 3. Define the loss function
        self.cel = nn.CrossEntropyLoss()
        
        # 4. Initialize the SAM optimizer
        self.optimizer = SAM(self.net.parameters(), torch.optim.SGD, lr=lr, momentum=0.9)

    def forward(self, x):
        """Defines the forward pass of the model."""
        return self.net(x)