"""
Configuration file for CLCS (Cluster-Level Confidence Learning for Segmentation)
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """모든 랜덤 시드 고정 (재현성 보장)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Paths ====================
# 학습/테스트 시 커맨드라인에서 지정 가능

# ==================== Training Config ====================
SEED = 42
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation

# ==================== Model Config ====================
# Backbone selection: "unet" or "unetr"
BACKBONE = "unet"  # Options: "unet", "unetr"

# Segmentation Network
IN_CHANNELS = 1  # CT/MRI (grayscale)
OUT_CHANNELS = 1  # Binary segmentation
SPATIAL_DIMS = 3  # 3D

# UNet specific
UNET_FEATURES = [32, 64, 128, 256, 512]  # nnU-Net style

# UNETR specific (Vision Transformer)
# ⚠️ UNETR은 고정 입력 크기 필요 - 데이터에 맞게 수정하세요!
# 예: 당신의 데이터가 (448, 448, 16)이면 이것도 (448, 448, 16)으로 변경
UNETR_IMG_SIZE = (128, 128, 128)  # UNETR 입력 크기 (patch size의 배수여야 함)
UNETR_FEATURE_SIZE = 16  # Feature size (img_size // patch_size)
UNETR_HIDDEN_SIZE = 768  # Transformer hidden size
UNETR_MLP_DIM = 3072  # MLP dimension
UNETR_NUM_HEADS = 12  # Number of attention heads

# Patch Classifier
# Auto-adaptive patch size: 자동으로 lesion 크기에 맞춰 조정
PATCH_SIZE_MODE = "auto"  # "auto" 또는 고정 크기 tuple (D, H, W)
# auto 모드: cluster 크기를 보고 자동으로 적절한 patch 크기 계산
# 고정 모드 예: (32, 96, 96)

# Auto patch size 설정
MAX_PATCH_SIZE = (64, 256, 256)  # 최대 patch 크기 제한
TARGET_PATCH_VOXELS = 128 * 128 * 32  # ~524K voxels (목표 크기)
MIN_PATCH_SIZE = (16, 32, 32)  # 최소 patch 크기

CLASSIFIER_FEATURES = [32, 64, 128]
DROPOUT_RATE = 0.3

# ==================== Training Hyperparameters ====================
MAX_EPOCHS = 200
BATCH_SIZE = 2  # 3D volume은 메모리 intensive
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Loss weights
LAMBDA_CLUSTER = 0.5  # L_total = L_seg + λ * L_cluster

# ==================== Data Augmentation ====================
# MONAI transforms에서 사용
# ⚠️ 당신의 실제 데이터 크기에 맞게 수정하세요!
SPATIAL_SIZE = [128, 128, 128]  # Crop/Pad to this size (예: [448, 448, 16] for your task)
INTENSITY_RANGE = (-1000, 1000)  # CT HU window

# ==================== Cluster Extraction ====================
PROB_THRESHOLD = 0.5  # Probability map threshold
MIN_CLUSTER_SIZE = 10  # Minimum voxels per cluster
IOU_THRESHOLD = 0.5  # IoU threshold for TP/FP labeling

# ==================== Patch Extraction ====================
CONTEXT_FACTOR = 2.0  # Bbox expansion factor (2x padding)

# ==================== Evaluation ====================
# FROC curve operating points
FROC_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0]

# ==================== Device ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# ==================== Checkpoint ====================
CHECKPOINT_DIR = "./checkpoints"
SAVE_EVERY_N_EPOCHS = 10

