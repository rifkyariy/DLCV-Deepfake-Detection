# model.py

import torch
from torch import nn
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
import os
import sys

# import src/utils 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from utils.sam import SAM # Assuming 'utils/sam.py' is in the same directory

class Detector(nn.Module):
    """
    A deepfake detector model using EfficientNetV2-M as the backbone
    and the Sharpness-Aware Minimization (SAM) optimizer.
    """

    def __init__(self, lr=0.001, num_classes=2):
        """
        Initializes the Detector model.

        Args:
            lr (float): The learning rate for the optimizer.
            num_classes (int): The number of output classes for the classifier.
        """
        super(Detector, self).__init__()
        
        # --- MODIFIED SECTION: Use the EfficientNetV2-M model ---
        # 1. Load the pre-trained EfficientNetV2-M model using the modern weights API
        weights = EfficientNet_V2_M_Weights.DEFAULT
        self.net = efficientnet_v2_m(weights=weights)
        # ------------------------------------------------------

        # 2. Replace the final classifier layer for your specific task
        # This logic remains the same as the classifier structure is consistent.
        in_features = self.net.classifier[1].in_features
        
        # Create a new sequential classifier with a linear layer for `num_classes`
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), # Slightly increased dropout for the larger model
            nn.Linear(in_features, num_classes)
        )
        
        # 3. Define the loss function
        self.cel = nn.CrossEntropyLoss()
        
        # 4. Initialize the SAM optimizer
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=lr, momentum=0.9)

    def forward(self, x):
        """Defines the forward pass of the model."""
        return self.net(x)
    
    def training_step(self, x, target):
        """
        Performs a single training step using the SAM optimizer.
        """
        for i in range(2):
            pred_cls = self(x)
            if i == 0:
                pred_first = pred_cls  

            loss = self.cel(pred_cls, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if i == 0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        return pred_first