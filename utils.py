"""
Utility functions for CLCS
- Cluster extraction (Stage 2)
- Adaptive patch extraction (Stage 3)
- IoU computation
- FROC curve calculation
"""
import torch
import numpy as np
from scipy import ndimage
from typing import List, Tuple, Dict
import torch.nn.functional as F

import config


# ==================== Stage 2: Cluster Extraction ====================

def extract_clusters(prob_map: np.ndarray, threshold: float = config.PROB_THRESHOLD) -> List[np.ndarray]:
    """
    Connected Component Analysis to extract individual clusters
    
    Args:
        prob_map: (D, H, W) probability map
        threshold: probability threshold
    
    Returns:
        clusters: List of binary masks for each cluster
    """
    # Threshold
    binary_mask = (prob_map > threshold).astype(np.int32)
    
    # Connected component analysis
    labeled_array, num_features = ndimage.label(binary_mask)
    
    clusters = []
    for i in range(1, num_features + 1):
        cluster_mask = (labeled_array == i)
        
        # Filter small clusters
        if np.sum(cluster_mask) >= config.MIN_CLUSTER_SIZE:
            clusters.append(cluster_mask)
    
    return clusters


def compute_bbox_with_context(cluster_mask: np.ndarray, context_factor: float = config.CONTEXT_FACTOR) -> Tuple:
    """
    Compute bounding box with context padding
    
    Args:
        cluster_mask: (D, H, W) binary mask
        context_factor: expansion factor (2.0 = 2x size)
    
    Returns:
        bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    coords = np.where(cluster_mask)
    
    if len(coords[0]) == 0:
        return None
    
    z_min, z_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()
    
    # Compute center and size
    z_center, y_center, x_center = (z_min + z_max) / 2, (y_min + y_max) / 2, (x_min + x_max) / 2
    z_size, y_size, x_size = z_max - z_min, y_max - y_min, x_max - x_min
    
    # Expand with context
    z_size_new = int(z_size * context_factor)
    y_size_new = int(y_size * context_factor)
    x_size_new = int(x_size * context_factor)
    
    # New bbox (clipped to volume bounds)
    volume_shape = cluster_mask.shape
    z_min_new = max(0, int(z_center - z_size_new / 2))
    z_max_new = min(volume_shape[0], int(z_center + z_size_new / 2))
    y_min_new = max(0, int(y_center - y_size_new / 2))
    y_max_new = min(volume_shape[1], int(y_center + y_size_new / 2))
    x_min_new = max(0, int(x_center - x_size_new / 2))
    x_max_new = min(volume_shape[2], int(x_center + x_size_new / 2))
    
    return (z_min_new, z_max_new, y_min_new, y_max_new, x_min_new, x_max_new)


# ==================== Stage 3: Adaptive Patch Extraction ====================

def compute_adaptive_patch_size(bbox_shape: tuple) -> tuple:
    """
    Cluster bbox 크기를 기반으로 적절한 patch size를 자동 계산
    Aspect ratio를 유지하면서 target voxels에 맞춤
    
    Args:
        bbox_shape: (D, H, W) bbox의 실제 크기
    
    Returns:
        patch_size: (D, H, W) 계산된 patch 크기
    """
    d, h, w = bbox_shape
    
    # 현재 voxels
    current_voxels = d * h * w
    
    # Target voxels에 맞춰 scale factor 계산
    if current_voxels > config.TARGET_PATCH_VOXELS:
        scale = (config.TARGET_PATCH_VOXELS / current_voxels) ** (1/3)
    else:
        scale = 1.0  # 이미 작으면 그대로 사용
    
    # Aspect ratio 유지하면서 스케일링
    patch_d = max(config.MIN_PATCH_SIZE[0], min(config.MAX_PATCH_SIZE[0], int(d * scale)))
    patch_h = max(config.MIN_PATCH_SIZE[1], min(config.MAX_PATCH_SIZE[1], int(h * scale)))
    patch_w = max(config.MIN_PATCH_SIZE[2], min(config.MAX_PATCH_SIZE[2], int(w * scale)))
    
    # 8의 배수로 맞춤 (GPU 최적화)
    patch_d = ((patch_d + 7) // 8) * 8
    patch_h = ((patch_h + 7) // 8) * 8
    patch_w = ((patch_w + 7) // 8) * 8
    
    return (patch_d, patch_h, patch_w)


def extract_adaptive_patch(
    image: torch.Tensor,
    prob_map: torch.Tensor,
    cluster_mask: np.ndarray,
    patch_size_mode: str = config.PATCH_SIZE_MODE
) -> torch.Tensor:
    """
    Extract and resize patch to adaptive size using adaptive pooling
    
    Args:
        image: (1, D, H, W) original image
        prob_map: (1, D, H, W) probability map
        cluster_mask: (D, H, W) binary cluster mask
        patch_size_mode: "auto" 또는 고정 크기 tuple (D, H, W)
    
    Returns:
        patch: (2, D, H, W) - [image, prob_map] with adaptive patch_size
    """
    bbox = compute_bbox_with_context(cluster_mask)
    
    if bbox is None:
        # Default size for empty cluster
        default_size = config.MIN_PATCH_SIZE if patch_size_mode == "auto" else patch_size_mode
        if isinstance(default_size, int):
            default_size = (default_size, default_size, default_size)
        return torch.zeros(2, *default_size, device=image.device)
    
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    
    # Auto patch size 계산
    if patch_size_mode == "auto":
        bbox_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
        patch_size = compute_adaptive_patch_size(bbox_shape)
    else:
        patch_size = patch_size_mode
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
    
    # Crop
    image_patch = image[:, z_min:z_max, y_min:y_max, x_min:x_max]
    prob_patch = prob_map[:, z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Adaptive pooling (supports any size)
    image_patch = F.adaptive_avg_pool3d(image_patch, patch_size)
    prob_patch = F.adaptive_avg_pool3d(prob_patch, patch_size)
    
    # Concatenate [image, prob_map]
    patch = torch.cat([image_patch, prob_patch], dim=0)  # (2, D, H, W)
    
    return patch


def extract_patches_batch(
    images: torch.Tensor,
    prob_maps: torch.Tensor,
    labels: torch.Tensor = None
) -> Tuple[List[torch.Tensor], torch.Tensor, List]:
    """
    Extract patches from a batch of images (auto-adaptive sizes)
    
    Args:
        images: (B, 1, D, H, W) input images
        prob_maps: (B, 1, D, H, W) probability maps
        labels: (B, 1, D, H, W) ground truth labels (optional, for training)
    
    Returns:
        patches: List of (2, D, H, W) patches (각 patch는 다른 크기 가능)
        cluster_labels: (N,) TP/FP labels (1=TP, 0=FP)
        cluster_info: List of cluster metadata (patch_size 포함)
    """
    batch_size = images.size(0)
    all_patches = []
    all_labels = []
    cluster_info = []
    
    for b in range(batch_size):
        image = images[b]  # (1, D, H, W)
        prob_map = prob_maps[b]  # (1, D, H, W)
        
        # Extract clusters from probability map
        prob_np = torch.sigmoid(prob_map[0]).detach().cpu().numpy()  # (D, H, W)
        clusters = extract_clusters(prob_np)
        
        for cluster_idx, cluster_mask in enumerate(clusters):
            # Extract patch (auto-adaptive size)
            patch = extract_adaptive_patch(image, prob_map, cluster_mask)
            all_patches.append(patch)
            
            # Compute cluster label (TP or FP) if ground truth is provided
            if labels is not None:
                gt_np = labels[b, 0].cpu().numpy()  # (D, H, W)
                iou = compute_iou(cluster_mask, gt_np > 0.5)
                label = 1.0 if iou > config.IOU_THRESHOLD else 0.0
                all_labels.append(label)
            
            cluster_info.append({
                'batch_idx': b,
                'cluster_idx': cluster_idx,
                'size': np.sum(cluster_mask),
                'patch_shape': patch.shape,  # (2, D, H, W)
                'bbox': compute_bbox_with_context(cluster_mask, context_factor=1.0)
            })
    
    if len(all_patches) == 0:
        # No clusters found
        return [], torch.zeros(0, device=images.device), []
    
    cluster_labels = torch.tensor(all_labels, device=images.device) if labels is not None else None
    
    return all_patches, cluster_labels, cluster_info


# ==================== IoU Computation ====================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks
    
    Args:
        mask1, mask2: binary masks (same shape)
    
    Returns:
        iou: IoU score [0, 1]
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


# ==================== FROC Curve Calculation ====================

def compute_froc(
    predictions: List[Dict],
    ground_truths: List[np.ndarray],
    confidence_thresholds: np.ndarray = None
) -> Dict:
    """
    Compute FROC (Free-Response ROC) curve
    
    Args:
        predictions: List of dicts with keys: 'clusters', 'confidences', 'case_id'
            - clusters: List of cluster masks
            - confidences: List of confidence scores
        ground_truths: List of ground truth masks
        confidence_thresholds: Thresholds to evaluate (default: 0.0 to 1.0)
    
    Returns:
        froc_data: Dict with 'sensitivities', 'fp_rates', 'thresholds'
    """
    if confidence_thresholds is None:
        confidence_thresholds = np.linspace(0, 1, 101)
    
    num_cases = len(predictions)
    num_gt_lesions = sum([len(extract_clusters(gt)) for gt in ground_truths])
    
    sensitivities = []
    fp_rates = []
    
    for threshold in confidence_thresholds:
        tp_count = 0
        fp_count = 0
        
        for pred, gt in zip(predictions, ground_truths):
            clusters = pred['clusters']
            confidences = pred['confidences']
            
            # Filter by confidence threshold
            filtered_clusters = [c for c, conf in zip(clusters, confidences) if conf > threshold]
            
            gt_clusters = extract_clusters(gt)
            matched_gts = set()
            
            for pred_cluster in filtered_clusters:
                # Check if this prediction matches any ground truth
                is_tp = False
                for gt_idx, gt_cluster in enumerate(gt_clusters):
                    if gt_idx not in matched_gts:
                        iou = compute_iou(pred_cluster, gt_cluster)
                        if iou > config.IOU_THRESHOLD:
                            is_tp = True
                            matched_gts.add(gt_idx)
                            break
                
                if is_tp:
                    tp_count += 1
                else:
                    fp_count += 1
        
        sensitivity = tp_count / num_gt_lesions if num_gt_lesions > 0 else 0.0
        fp_rate = fp_count / num_cases if num_cases > 0 else 0.0
        
        sensitivities.append(sensitivity)
        fp_rates.append(fp_rate)
    
    return {
        'sensitivities': np.array(sensitivities),
        'fp_rates': np.array(fp_rates),
        'thresholds': confidence_thresholds
    }


def get_sensitivity_at_fp_rate(froc_data: Dict, target_fp_rate: float) -> float:
    """
    Get sensitivity at a specific FP rate (interpolated)
    
    Args:
        froc_data: Output from compute_froc()
        target_fp_rate: Target FP rate (e.g., 0.5 FP/case)
    
    Returns:
        sensitivity: Sensitivity at target FP rate
    """
    fp_rates = froc_data['fp_rates']
    sensitivities = froc_data['sensitivities']
    
    # Interpolate
    return np.interp(target_fp_rate, fp_rates, sensitivities)


if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test cluster extraction
    test_prob = np.random.rand(64, 64, 64)
    test_prob[20:30, 20:30, 20:30] = 0.9  # Simulated lesion
    clusters = extract_clusters(test_prob, threshold=0.5)
    print(f"✓ Found {len(clusters)} clusters")
    
    # Test patch extraction
    if len(clusters) > 0:
        test_image = torch.randn(1, 64, 64, 64)
        test_prob_map = torch.tensor(test_prob).unsqueeze(0).float()
        patch = extract_adaptive_patch(test_image, test_prob_map, clusters[0])
        print(f"✓ Extracted patch shape: {patch.shape}")
    
    print("✓ All tests passed!")

