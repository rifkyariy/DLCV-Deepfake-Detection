# model.py

import torch
from torch import nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import os
import sys


# import src/utils 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from utils.sam import SAM # Assuming 'utils/sam.py' is in the same directory

class Detector(nn.Module):
    """
    A deepfake detector model using EfficientNetV2-S as the backbone
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
        
        # 1. Load the pre-trained EfficientNetV2-S model using the modern weights API
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.net = efficientnet_v2_s(weights=weights)

        # 2. Replace the final classifier layer for your specific task
        # Get the number of input features from the original classifier
        in_features = self.net.classifier[1].in_features
        
        # Create a new sequential classifier with a linear layer for `num_classes`
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        # 3. Define the loss function
        self.cel = nn.CrossEntropyLoss()
        
        # 4. Initialize the SAM optimizer, which wraps a base optimizer (SGD)
        # The training script in train.py will access this `optimizer` attribute.
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=lr, momentum=0.9)

    def forward(self, x):
        """Defines the forward pass of the model."""
        return self.net(x)
    
    def training_step(self, x, target):
        """
        Performs a single training step using the SAM optimizer.
        This method is called directly from your train.py loop.
        """
        # SAM requires two forward/backward passes to find a parameter space
        # with uniformly low loss (i.e., a "flatter" minimum).
        for i in range(2):
            pred_cls = self(x)
            if i == 0:
                # Save the predictions from the first step (before the perturbation)
                pred_first = pred_cls  

            loss = self.cel(pred_cls, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if i == 0:
                self.optimizer.first_step(zero_grad=True)  # Ascent step to find a point with high sharpness
            else:
                self.optimizer.second_step(zero_grad=True) # Descent step at the new perturbed point
        
        # Return the predictions from before the second "ascent" step for metric calculation
        return pred_first