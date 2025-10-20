import torch
from torch import nn
import timm
from utils.sam import SAM

class Detector(nn.Module):
    """
    A deepfake detector model using MobileViTv2-2.0 as the backbone
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
        
        print("[INFO] Loading MobileViTv2-2.0 using timm...")
        
        # Load MobileViTv2-2.0 with custom pretrained weights from Apple
        pretrained_url = "https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/384x384/mobilevitv2-2.0.pt"
        
        try:
            # Create model without pretrained weights first
            self.net = timm.create_model(
                'mobilevitv2_200',
                pretrained=False,
                num_classes=1000
            )
            
            # Download and load Apple's pretrained weights
            state_dict = torch.hub.load_state_dict_from_url(
                pretrained_url,
                map_location='cpu',
                progress=True
            )
            
            # Load the weights
            self.net.load_state_dict(state_dict, strict=False)
            print("[INFO] Loaded Apple's pretrained weights")
            
        except Exception as e:
            print(f"[WARN] Could not load Apple weights: {e}")
            print("[INFO] Using timm's default pretrained weights")
            self.net = timm.create_model(
                'mobilevitv2_200',
                pretrained=True,
                num_classes=1000
            )
        
        # Get the number of input features for the classifier
        in_features = self.net.get_classifier().in_features
        
        # Replace classifier for binary classification
        # Don't use reset_classifier, manually replace the head
        if hasattr(self.net, 'head'):
            if hasattr(self.net.head, 'fc'):
                # MobileViTv2 has head.fc structure
                self.net.head.fc = nn.Sequential(
                    nn.Dropout(p=0.3, inplace=True),
                    nn.Linear(in_features, num_classes)
                )
            else:
                # Direct head replacement
                self.net.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(p=0.3, inplace=True),
                    nn.Linear(in_features, num_classes)
                )
        else:
            # Fallback: use classifier attribute
            self.net.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        
        print(f"[INFO] Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
        
        # Define the loss function
        self.cel = nn.CrossEntropyLoss()
        
        # Initialize the SAM optimizer
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(
            self.parameters(),
            base_optimizer,
            lr=lr,
            momentum=0.9
        )

    def forward(self, x):
        """Defines the forward pass of the model."""
        return self.net(x)