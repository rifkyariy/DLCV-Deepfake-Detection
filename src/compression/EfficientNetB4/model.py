import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

import os
import sys
# Ensure src is in sys.path for Docker/project structure
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    
from src.utils.sam import SAM

class Detector(nn.Module):

    def __init__(self, lr=0.001):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)
        self.cel = nn.CrossEntropyLoss()
        
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=lr, momentum=0.9)

    def forward(self, x):
        x = self.net(x)
        return x
    
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

