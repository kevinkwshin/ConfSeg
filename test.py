"""
Test script for CLCS
Usage:
    python test.py --image_dir /path/to/test/images --label_dir /path/to/test/labels --checkpoint /path/to/model.pth
"""
import os
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from model import create_model
from train import get_data_dicts, get_transforms
from utils import (
    extract_clusters, 
    extract_patches_batch,
    compute_froc,
    get_sensitivity_at_fp_rate
)


# ==================== Inference ====================

def predict_single_case(
    model: torch.nn.Module,
    image: torch.Tensor
) -> tuple:
    """
    Predict segmentation and cluster confidences for a single case
    
    Args:
        model: CLCS model
        image: (1, 1, D, H, W) input image
    
    Returns:
        seg_prob: (D, H, W) segmentation probability map
        clusters: List of cluster masks
        confidences: List of confidence scores for each cluster
    """
    model.eval()
    
    with torch.no_grad():
        # Segmentation
        seg_logits = model.forward_segmentation(image)
        seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()  # (D, H, W)
        
        # Extract clusters
        clusters = extract_clusters(seg_prob, threshold=config.PROB_THRESHOLD)
        
        if len(clusters) == 0:
            return seg_prob, [], []
        
        # Extract patches for each cluster & predict confidences
        confidences = []
        for cluster_mask in clusters:
            from utils import extract_adaptive_patch
            patch = extract_adaptive_patch(
                image[0],  # (1, D, H, W)
                seg_logits[0],  # (1, D, H, W)
                cluster_mask
            )
            # Predict confidence for this patch
            conf = model.forward_classifier(patch.unsqueeze(0)).cpu().item()  # scalar
            confidences.append(conf)
        
        confidences = np.array(confidences)  # (N,)
    
    return seg_prob, clusters, confidences


def test_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    compute_froc_curve: bool = True
) -> Dict:
    """
    Evaluate model on test set
    
    Args:
        model: CLCS model
        test_loader: Test data loader
        compute_froc_curve: Whether to compute FROC curve
    
    Returns:
        results: Dictionary with metrics
    """
    model.eval()
    
    # Metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    # For FROC computation
    all_predictions = []
    all_ground_truths = []
    
    # Per-case results
    case_results = []
    
    print("\nRunning inference...")
    for idx, batch_data in enumerate(tqdm(test_loader)):
        image = batch_data["image"].to(config.DEVICE)  # (B, 1, D, H, W)
        label = batch_data["label"].to(config.DEVICE)  # (B, 1, D, H, W)
        
        batch_size = image.size(0)
        
        for b in range(batch_size):
            # Single case prediction
            seg_prob, clusters, confidences = predict_single_case(
                model,
                image[b:b+1]
            )
            
            # Compute Dice score
            seg_binary = (seg_prob > 0.5).astype(np.float32)
            label_np = label[b, 0].cpu().numpy()
            
            dice_score = np.sum(seg_binary * label_np) * 2.0 / (np.sum(seg_binary) + np.sum(label_np) + 1e-8)
            
            # Store results
            case_results.append({
                'case_id': idx * batch_size + b,
                'dice': dice_score,
                'num_clusters': len(clusters),
                'confidences': confidences
            })
            
            # For FROC
            if compute_froc_curve:
                all_predictions.append({
                    'clusters': clusters,
                    'confidences': confidences,
                    'case_id': idx * batch_size + b
                })
                all_ground_truths.append(label_np)
    
    # Compute overall metrics
    avg_dice = np.mean([r['dice'] for r in case_results])
    
    results = {
        'avg_dice': avg_dice,
        'case_results': case_results
    }
    
    # Compute FROC curve
    if compute_froc_curve and len(all_predictions) > 0:
        print("\nComputing FROC curve...")
        froc_data = compute_froc(all_predictions, all_ground_truths)
        results['froc'] = froc_data
        
        # Compute sensitivity at key operating points
        print("\n=== FROC Results ===")
        for fp_rate in config.FROC_FP_RATES:
            sensitivity = get_sensitivity_at_fp_rate(froc_data, fp_rate)
            print(f"Sensitivity @ {fp_rate} FP/case: {sensitivity:.4f}")
            results[f'sens_at_{fp_rate}_fp'] = sensitivity
    
    return results


# ==================== Visualization ====================

def plot_froc_curve(froc_data: Dict, save_path: str = None):
    """
    Plot FROC curve
    
    Args:
        froc_data: Output from compute_froc()
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(froc_data['fp_rates'], froc_data['sensitivities'], 'b-', linewidth=2)
    plt.xlabel('False Positives per Case', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.title('FROC Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Mark key operating points
    for fp_rate in config.FROC_FP_RATES:
        sens = get_sensitivity_at_fp_rate(froc_data, fp_rate)
        plt.plot(fp_rate, sens, 'ro', markersize=8)
        plt.text(fp_rate, sens + 0.02, f'{sens:.3f}', ha='center', fontsize=10)
    
    plt.xlim([0, max(config.FROC_FP_RATES) + 0.5])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"FROC curve saved to {save_path}")
    else:
        plt.show()


# ==================== Main Test Function ====================

def test(
    image_dir: str,
    label_dir: str,
    checkpoint_path: str,
    output_dir: str = "./results"
):
    """
    Main test function
    
    Args:
        image_dir: Directory containing test images
        label_dir: Directory containing test labels
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save results
    """
    # Set seed
    config.set_seed(config.SEED)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading test data...")
    data_dicts = get_data_dicts(image_dir, label_dir)
    
    if len(data_dicts) == 0:
        raise ValueError(f"No test data found in {image_dir} and {label_dir}")
    
    test_dataset = Dataset(data=data_dicts, transform=get_transforms(is_train=False))
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one case at a time
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    # Get backbone from checkpoint (fallback to config if not found)
    backbone = checkpoint.get('backbone', config.BACKBONE)
    print(f"Detected backbone: {backbone.upper()}")
    
    # Create model with correct backbone
    original_backbone = config.BACKBONE
    config.BACKBONE = backbone
    model = create_model()
    config.BACKBONE = original_backbone
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Test
    results = test_model(model, test_loader, compute_froc_curve=True)
    
    # Print results
    print("\n" + "="*50)
    print("=== Test Results ===")
    print("="*50)
    print(f"Average Dice Score: {results['avg_dice']:.4f}")
    
    if 'froc' in results:
        print("\nFROC Analysis:")
        for fp_rate in config.FROC_FP_RATES:
            key = f'sens_at_{fp_rate}_fp'
            if key in results:
                print(f"  Sensitivity @ {fp_rate} FP/case: {results[key]:.4f}")
    
    # Save FROC curve
    if 'froc' in results:
        froc_path = os.path.join(output_dir, "froc_curve.png")
        plot_froc_curve(results['froc'], save_path=froc_path)
    
    # Save results to file
    import json
    results_path = os.path.join(output_dir, "test_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    results_to_save = {
        'avg_dice': float(results['avg_dice']),
        'num_cases': len(results['case_results'])
    }
    
    if 'froc' in results:
        for fp_rate in config.FROC_FP_RATES:
            key = f'sens_at_{fp_rate}_fp'
            if key in results:
                results_to_save[key] = float(results[key])
    
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    print("="*50)


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Test CLCS model")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to test images")
    parser.add_argument("--label_dir", type=str, required=True, help="Path to test labels")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    
    args = parser.parse_args()
    
    test(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

