import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

import os
import sys
# Ensure src is in sys.path for Docker/project structure
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    
from src.utils.sam import SAM

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

    def __init__(self, lr=0.001):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)
        self.cel = nn.CrossEntropyLoss()
        
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=lr, momentum=0.9)

    def forward(self, x):
        x = self.net(x)
        return x
    
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
        # This part of your logic is correct for SAM
        for i in range(2):
            pred_cls = self(x)
            if i == 0:
                pred_first = pred_cls
            loss_cls = self.cel(pred_cls, target)
            loss = loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i == 0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        return pred_first