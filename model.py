"""
Model architecture for CLCS
- Segmentation Network (MONAI UNet or UNETR)
- Patch Classifier (3D CNN for TP/FP classification)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet, UNETR
from typing import Tuple

import config


class PatchClassifier3D(nn.Module):
    """
    3D CNN classifier for patch-level TP/FP classification
    
    Supports both isotropic and anisotropic input patches:
    - Isotropic: (B, 2, 32, 32, 32)
    - Anisotropic: (B, 2, D, H, W) where D!=H!=W
    
    Uses adaptive pooling to handle various input sizes.
    
    Input: (B, 2, D, H, W) patches [image, prob_map]
    Output: (B, 1) confidence scores [0, 1]
    """
    
    def __init__(
        self,
        in_channels: int = 2,  # Image + Probability map
        features: list = config.CLASSIFIER_FEATURES,
        dropout: float = config.DROPOUT_RATE
    ):
        super().__init__()
        
        layers = []
        channels = [in_channels] + features
        
        # Convolutional layers with residual connections
        for i in range(len(features)):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(channels[i], channels[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm3d(channels[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(channels[i+1], channels[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm3d(channels[i+1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2)
                )
            )
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        # Fully connected layers
        fc_input_size = features[-1] * 2 * 2 * 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Confidence score [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, D, H, W) - concatenated [image, prob_map]
               Supports both isotropic (D=H=W) and anisotropic patches
        Returns:
            confidence: (B, 1) - TP/FP confidence scores
        """
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        confidence = self.fc(x)
        return confidence


class CLCS_Model(nn.Module):
    """
    Complete CLCS model combining:
    1. Segmentation Network (UNet or UNETR)
    2. Patch Classifier (for cluster confidence)
    """
    
    def __init__(
        self,
        backbone: str = config.BACKBONE,
        in_channels: int = config.IN_CHANNELS,
        out_channels: int = config.OUT_CHANNELS,
        spatial_dims: int = config.SPATIAL_DIMS,
        unet_features: list = config.UNET_FEATURES,
        unetr_img_size: tuple = config.UNETR_IMG_SIZE,
        unetr_feature_size: int = config.UNETR_FEATURE_SIZE,
        unetr_hidden_size: int = config.UNETR_HIDDEN_SIZE,
        unetr_mlp_dim: int = config.UNETR_MLP_DIM,
        unetr_num_heads: int = config.UNETR_NUM_HEADS
    ):
        super().__init__()
        
        self.backbone = backbone.lower()
        
        # Stage 1: Segmentation Network - Select backbone
        if self.backbone == "unet":
            print(f"Using UNet backbone")
            self.segmentation_net = UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=unet_features,
                strides=[2, 2, 2, 2],
                num_res_units=2,
                norm="batch"
            )
        elif self.backbone == "unetr":
            print(f"Using UNETR backbone (Vision Transformer)")
            self.segmentation_net = UNETR(
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=unetr_img_size,
                feature_size=unetr_feature_size,
                hidden_size=unetr_hidden_size,
                mlp_dim=unetr_mlp_dim,
                num_heads=unetr_num_heads,
                pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.0
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose 'unet' or 'unetr'")
        
        # Stage 4: Patch Classifier
        self.patch_classifier = PatchClassifier3D(
            in_channels=in_channels + out_channels  # Image + Prob map
        )
    
    def forward_segmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stage 1: Segmentation only
        Args:
            x: (B, 1, D, H, W) input image
        Returns:
            logits: (B, 1, D, H, W) segmentation logits
        """
        return self.segmentation_net(x)
    
    def forward_classifier(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Stage 4: Patch classification only
        Args:
            patches: (N, 2, 32, 32, 32) - N patches from all clusters
        Returns:
            confidences: (N, 1) - cluster confidence scores
        """
        return self.patch_classifier(patches)
    
    def forward(
        self, 
        x: torch.Tensor, 
        patches: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass (training mode)
        
        Args:
            x: (B, 1, D, H, W) input images
            patches: (N, 2, 32, 32, 32) extracted patches (optional, for training)
        
        Returns:
            seg_logits: (B, 1, D, H, W) segmentation logits
            cluster_confidences: (N, 1) cluster confidences (if patches provided)
        """
        # Stage 1: Segmentation
        seg_logits = self.forward_segmentation(x)
        
        # Stage 4: Cluster confidence (if patches provided)
        cluster_confidences = None
        if patches is not None and patches.size(0) > 0:
            cluster_confidences = self.forward_classifier(patches)
        
        return seg_logits, cluster_confidences


def create_model() -> CLCS_Model:
    """Factory function to create CLCS model"""
    model = CLCS_Model(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        spatial_dims=config.SPATIAL_DIMS,
        unet_features=config.UNET_FEATURES
    )
    return model.to(config.DEVICE)


if __name__ == "__main__":
    # Test model creation
    print("Testing CLCS Model...")
    config.set_seed(config.SEED)
    
    model = create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_image = torch.randn(2, 1, 64, 64, 64).to(config.DEVICE)
    dummy_patches = torch.randn(5, 2, 32, 32, 32).to(config.DEVICE)
    
    seg_out, conf_out = model(dummy_image, dummy_patches)
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Cluster confidence shape: {conf_out.shape}")
    print("✓ Model test passed!")

