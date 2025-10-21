import torch
from torch import nn
import torchvision
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

# Adjust sys.path to include 'src' directory ---
import sys
import os

# Add project's 'src' directory to the path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir)) 
sys.path.insert(0, src_dir)
# -------------------------

from utils.sam import SAM 

def decompress_state_dict(compressed_dict):
    """
    Decompress the state dictionary back to float32
    Handles both compressed and regular state dicts
    """
    state_dict = {}
    
    for key, value in compressed_dict.items():
        if isinstance(value, dict) and 'quantized' in value:
            # Dequantize
            quantized = value['quantized']
            min_val = value['min']
            scale = value['scale']
            
            decompressed = quantized.float() * scale + min_val
            state_dict[key] = decompressed.reshape(value['shape'])
        else:
            # Convert float16 back to float32 if needed
            if isinstance(value, torch.Tensor) and value.dtype == torch.float16:
                state_dict[key] = value.float()
            else:
                state_dict[key] = value
    
    return state_dict

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
        
        # 1. Load the pre-trained EfficientNetV2-M model using the modern weights API
        weights = EfficientNet_V2_M_Weights.DEFAULT
        self.net = efficientnet_v2_m(weights=weights)

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
    
    def load_compressed_state_dict(self, state_dict_path, strict=True):
        """
        Load state dict from either compressed or regular checkpoint
        
        Args:
            state_dict_path: Path to .pth file
            strict: Whether to strictly enforce that the keys match
        """
        # Load the state dict
        checkpoint = torch.load(state_dict_path, map_location='cpu')
        
        # Check if it's compressed
        is_compressed = False
        for key, value in checkpoint.items():
            if isinstance(value, dict) and 'quantized' in value:
                is_compressed = True
                break
        
        # Decompress if needed
        if is_compressed:
            print("Detected compressed checkpoint, decompressing...")
            state_dict = decompress_state_dict(checkpoint)
        else:
            state_dict = checkpoint
        
        # Load into model
        return self.load_state_dict(state_dict, strict=strict)
    
    def training_step(self, x, target):
        """
        Performs a single training step using the SAM optimizer.
        This method is called directly from your train.py loop.
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