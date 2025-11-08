"""
Training script for CLCS
Usage:
    python train.py --image_dir /path/to/images --label_dir /path/to/labels
"""
import os
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from monai.data import Dataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, RandCropByPosNegLabeld, RandFlipd,
    RandRotate90d, EnsureTyped, ToTensord, DivisiblePadd, SpatialPadd
)
from monai.losses import DiceLoss
from tqdm import tqdm

import config
from model import create_model
from utils import extract_patches_batch


# ==================== Data Loading ====================

def get_data_dicts(image_dir: str, label_dir: str) -> list:
    """
    Create data dictionaries for MONAI Dataset
    
    Args:
        image_dir: Directory containing images (.nii.gz, .nii, .npy)
        label_dir: Directory containing labels (same filenames)
    
    Returns:
        data_dicts: List of {"image": path, "label": path}
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    # Support multiple formats
    extensions = ['*.nii.gz', '*.nii', '*.npy']
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(image_dir.glob(ext)))
    
    data_dicts = []
    for img_path in image_files:
        label_path = label_dir / img_path.name
        
        if not label_path.exists():
            # Try alternative extensions
            for ext in ['.nii.gz', '.nii', '.npy']:
                label_path = label_dir / (img_path.stem.replace('.nii', '') + ext)
                if label_path.exists():
                    break
        
        if label_path.exists():
            data_dicts.append({
                "image": str(img_path),
                "label": str(label_path)
            })
    
    print(f"Found {len(data_dicts)} image-label pairs")
    return data_dicts


def get_transforms(is_train: bool = True):
    """Create MONAI transforms for data preprocessing"""
    
    if is_train:
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.INTENSITY_RANGE[0],
                a_max=config.INTENSITY_RANGE[1],
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            # DivisiblePad: 이미지를 16의 배수로 패딩 (UNet의 4번 downsampling: 2^4=16)
            DivisiblePadd(keys=["image", "label"], k=16, mode="constant"),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            EnsureTyped(keys=["image", "label"])
        ])
    else:
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.INTENSITY_RANGE[0],
                a_max=config.INTENSITY_RANGE[1],
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            # DivisiblePad: validation에서도 동일하게 적용
            DivisiblePadd(keys=["image", "label"], k=16, mode="constant"),
            EnsureTyped(keys=["image", "label"])
        ])


def create_data_loaders(
    image_dir: str,
    label_dir: str,
    train_split: float = config.TRAIN_VAL_SPLIT
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders with automatic split
    
    Args:
        image_dir: Directory containing training images
        label_dir: Directory containing training labels
        train_split: Fraction of data for training (e.g., 0.8)
    
    Returns:
        train_loader, val_loader
    """
    # Get data dictionaries
    data_dicts = get_data_dicts(image_dir, label_dir)
    
    if len(data_dicts) == 0:
        raise ValueError(f"No data found in {image_dir} and {label_dir}")
    
    # Split into train/val
    total_size = len(data_dicts)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    # Random split
    train_dicts, val_dicts = random_split(
        data_dicts,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    print(f"Train: {len(train_dicts)} samples, Val: {len(val_dicts)} samples")
    
    # Create datasets
    train_dataset = Dataset(data=list(train_dicts), transform=get_transforms(is_train=True))
    val_dataset = Dataset(data=list(val_dicts), transform=get_transforms(is_train=False))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ==================== Loss Functions ====================

def compute_loss(
    seg_logits: torch.Tensor,
    labels: torch.Tensor,
    cluster_confidences: torch.Tensor,
    cluster_labels: torch.Tensor,
    dice_loss_fn: nn.Module,
    bce_loss_fn: nn.Module
) -> Tuple[torch.Tensor, dict]:
    """
    Compute total loss: L_total = L_seg + λ * L_cluster
    
    Args:
        seg_logits: (B, 1, D, H, W) segmentation logits
        labels: (B, 1, D, H, W) ground truth labels
        cluster_confidences: (N, 1) cluster confidence scores
        cluster_labels: (N,) cluster ground truth (TP=1, FP=0)
        dice_loss_fn: Dice loss function
        bce_loss_fn: Binary cross-entropy loss
    
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual losses
    """
    # Segmentation loss
    seg_probs = torch.sigmoid(seg_logits)
    dice_loss = dice_loss_fn(seg_probs, labels)
    bce_loss = bce_loss_fn(seg_logits, labels.float())
    seg_loss = dice_loss + bce_loss
    
    # Cluster loss (if clusters exist)
    cluster_loss = torch.tensor(0.0, device=seg_logits.device)
    if cluster_confidences is not None and cluster_labels is not None and len(cluster_labels) > 0:
        cluster_loss = bce_loss_fn(cluster_confidences, cluster_labels.unsqueeze(1).float())
    
    # Total loss
    total_loss = seg_loss + config.LAMBDA_CLUSTER * cluster_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'seg': seg_loss.item(),
        'cluster': cluster_loss.item()
    }
    
    return total_loss, loss_dict


# ==================== Training Loop ====================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dice_loss_fn: nn.Module,
    bce_loss_fn: nn.Module,
    epoch: int
) -> dict:
    """Train for one epoch"""
    model.train()
    
    total_losses = {'total': 0, 'seg': 0, 'cluster': 0}
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_data in pbar:
        images = batch_data["image"].to(config.DEVICE)  # (B, 1, D, H, W)
        labels = batch_data["label"].to(config.DEVICE)  # (B, 1, D, H, W)
        
        optimizer.zero_grad()
        
        # Forward: Segmentation
        seg_logits = model.forward_segmentation(images)
        
        # Extract patches from predicted clusters
        with torch.no_grad():
            seg_probs = torch.sigmoid(seg_logits)
            patches, cluster_labels, _ = extract_patches_batch(images, seg_logits, labels)
        
        # Forward: Cluster classification (각 patch 개별 처리)
        cluster_confidences = None
        if len(patches) > 0:
            confidences = []
            for patch in patches:
                conf = model.forward_classifier(patch.unsqueeze(0))  # (1, 2, D, H, W) → (1, 1)
                confidences.append(conf)
            cluster_confidences = torch.cat(confidences, dim=0)  # (N, 1)
        
        # Compute loss
        loss, loss_dict = compute_loss(
            seg_logits, labels,
            cluster_confidences, cluster_labels,
            dice_loss_fn, bce_loss_fn
        )
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses.keys():
            total_losses[key] += loss_dict[key]
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'seg': f"{loss_dict['seg']:.4f}",
            'cls': f"{loss_dict['cluster']:.4f}"
        })
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    dice_loss_fn: nn.Module,
    bce_loss_fn: nn.Module
) -> dict:
    """Validate model"""
    model.eval()
    
    total_losses = {'total': 0, 'seg': 0, 'cluster': 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation"):
            images = batch_data["image"].to(config.DEVICE)
            labels = batch_data["label"].to(config.DEVICE)
            
            # Forward
            seg_logits = model.forward_segmentation(images)
            seg_probs = torch.sigmoid(seg_logits)
            
            # Extract patches
            patches, cluster_labels, _ = extract_patches_batch(images, seg_logits, labels)
            
            # Cluster classification (각 patch 개별 처리)
            cluster_confidences = None
            if len(patches) > 0:
                confidences = []
                for patch in patches:
                    conf = model.forward_classifier(patch.unsqueeze(0))
                    confidences.append(conf)
                cluster_confidences = torch.cat(confidences, dim=0)
            
            # Compute loss
            _, loss_dict = compute_loss(
                seg_logits, labels,
                cluster_confidences, cluster_labels,
                dice_loss_fn, bce_loss_fn
            )
            
            for key in total_losses.keys():
                total_losses[key] += loss_dict[key]
            num_batches += 1
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


# ==================== Main Training Function ====================

def train(
    image_dir: str,
    label_dir: str,
    max_epochs: int = config.MAX_EPOCHS,
    checkpoint_dir: str = config.CHECKPOINT_DIR,
    backbone: str = config.BACKBONE
):
    """
    Main training function
    
    Args:
        image_dir: Directory containing training images
        label_dir: Directory containing training labels
        max_epochs: Maximum number of epochs
        checkpoint_dir: Directory to save checkpoints
        backbone: Backbone network ("unet" or "unetr")
    """
    # Set seed
    config.set_seed(config.SEED)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(image_dir, label_dir)
    
    # Create model with specified backbone
    print(f"Creating model with {backbone.upper()} backbone...")
    # Temporarily override config
    original_backbone = config.BACKBONE
    config.BACKBONE = backbone
    model = create_model()
    config.BACKBONE = original_backbone
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs
    )
    
    # Loss functions
    dice_loss_fn = DiceLoss(sigmoid=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {max_epochs} epochs...")
    for epoch in range(1, max_epochs + 1):
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer,
            dice_loss_fn, bce_loss_fn, epoch
        )
        
        # Validate
        val_losses = validate(model, val_loader, dice_loss_fn, bce_loss_fn)
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{max_epochs}")
        print(f"Train Loss: {train_losses['total']:.4f} (seg: {train_losses['seg']:.4f}, cls: {train_losses['cluster']:.4f})")
        print(f"Val Loss:   {val_losses['total']:.4f} (seg: {val_losses['seg']:.4f}, cls: {val_losses['cluster']:.4f})")
        
        # Save checkpoint
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'backbone': backbone  # Save backbone info
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'backbone': backbone  # Save backbone info
            }, checkpoint_path)
    
    print(f"\n✓ Training completed! Best val loss: {best_val_loss:.4f}")


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Train CLCS model")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--label_dir", type=str, required=True, help="Path to training labels")
    parser.add_argument("--max_epochs", type=int, default=config.MAX_EPOCHS, help="Maximum epochs")
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR, help="Checkpoint directory")
    parser.add_argument("--backbone", type=str, default=config.BACKBONE, choices=["unet", "unetr"], 
                        help="Backbone network (default: unet)")
    
    args = parser.parse_args()
    
    train(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        max_epochs=args.max_epochs,
        checkpoint_dir=args.checkpoint_dir,
        backbone=args.backbone
    )


if __name__ == "__main__":
    main()

