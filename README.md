# CLCS: Cluster-Level Confidence Learning for Segmentation

Medical image segmentation with per-lesion confidence scores via TP/FP classification. (ConfSeg)

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python train.py --image_dir ./data/train/images --label_dir ./data/train/labels

# Test
python test.py --image_dir ./data/test/images --label_dir ./data/test/labels --checkpoint ./checkpoints/best_model.pth
```

## 핵심 기능

- **Auto-Adaptive Patch Size**: 어떤 input size (448×448×16 등)에도 자동 최적화
- **End-to-end Learning**: Segmentation + Classifier 동시 학습
- **FROC Evaluation**: False Positive rate 중심 평가
- **Zero Configuration**: Lesion 크기 분석 불필요

## 설정 (config.py)

```python
# Data
SPATIAL_SIZE = [128, 128, 128]  # 데이터 크기에 맞게 변경

# Model
BACKBONE = "unet"  # or "unetr"

# Patch Size (Auto)
PATCH_SIZE_MODE = "auto"  # 자동 최적화
MAX_PATCH_SIZE = (64, 256, 256)
TARGET_PATCH_VOXELS = 128 * 128 * 32

# Training
MAX_EPOCHS = 200
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
LAMBDA_CLUSTER = 0.5  # Cluster loss weight
```

## 파일 구조

```
CLCS/
├── config.py      # 설정
├── model.py       # UNet/UNETR + Classifier
├── utils.py       # Cluster extraction, FROC
├── train.py       # 학습 (auto train/val split)
├── test.py        # 테스트 + FROC evaluation
└── README.md      # 이 파일
```

## 작동 방식

```
Input Image
    ↓
Segmentation (UNet/UNETR) → Probability Map
    ↓
Cluster Extraction (CCA) → Individual Lesions
    ↓
Adaptive Patch Extraction → Fixed-size patches
    ↓
Patch Classifier → TP/FP Confidence [0,1]
    ↓
Output: Cluster Confidences
```

## 평가 지표

- **Dice Score**: Segmentation quality
- **FROC Curve**: Sensitivity vs FP/case
- **Cluster Confidence**: TP/FP classification AUC

## 예상 성능

| Metric | Target |
|--------|--------|
| Dice Score | >0.80 |
| FROC @ 0.5 FP/case | >0.85 |
| Cluster AUC | >0.90 |

## Citation

```
CLCS: Cluster-level Confidence Learning for Medical Image Segmentation
End-to-end framework for per-lesion confidence via TP/FP classification
```

## License

Research use only.
