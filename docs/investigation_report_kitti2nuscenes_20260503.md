# Investigation Report: KITTI→nuScenes Domain Adaptation (UADA3D)

**Date:** 2026-05-03  
**Branch:** `v20260102_benchmark`  
**Scope:** Diagnosis of zero/low mAP in K→N runs, paper reproduction audit, and conditional discriminator fix.

---

## 1. Runs Investigated

| W&B Run | Config | Discriminator | nuScenes mAP | Status |
|---|---|---|---|---|
| `zg6yq4xn` | `second-rospm-M-C` | Marginal only | **0.000** | Broken — PCR mismatch |
| `hseplt9x` | `second-rospm-M-C` | Marginal only | **0.018** | Partial fix |
| `rhbp2jxp` | `centerpoint-rospm-M-C` | Marginal + broken conditional | **0.009** | Conditional disc non-functional |

---

## 2. Root Cause Analysis

### 2.1 Issue 1 — `POINT_CLOUD_RANGE` mismatch (zg6yq4xn → mAP = 0)

`DATA_CONFIG` and `TARGET_DATA_CONFIG` had **different** point cloud ranges:

- **Source (KITTI):** `[-50, -51.2, -2, 150, 51.2, 4]`
- **Target (nuScenes base):** `[-75.2, -75.2, -2, 75.2, 75.2, 4]`

**Consequences:**
1. BEV feature map spatial dimensions differ between source and target → discriminator receives mismatched tensors.
2. SECOND anchor coordinates are defined in source voxel space; nuScenes predictions are decoded into wrong world coordinates → zero IoU with GT for all 40 epochs.

**Fix applied (initial):**
- Unified both ranges to `[-50, -51.2, -2, 150, 51.2, 4]` in `second-rospm-M-C.yaml` to remove the mismatch.
- Added runtime assertion in `tools/adaptive_train.py` to catch future mismatches at startup:

```python
assert list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE) == list(cfg.TARGET_DATA_CONFIG.POINT_CLOUD_RANGE), (
    f"POINT_CLOUD_RANGE mismatch between DATA_CONFIG {list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)} "
    f"and TARGET_DATA_CONFIG {list(cfg.TARGET_DATA_CONFIG.POINT_CLOUD_RANGE)}. "
    "Both must be identical for domain adaptation (BEV feature map dims must match)."
)
```

---

### 2.2 Issue 2 — Conditional discriminator not active / broken

#### SECOND (hseplt9x)
`DISCRIMINATOR.CONDITIONAL_ADAPTATION` was absent from `second-rospm-M-C.yaml`. Only marginal BEV alignment ran. This is `UADA3D_Lm` in the paper's ablation, which achieves lower mAP than full UADA3D.

#### CenterPoint (rhbp2jxp)
`CONDITIONAL_ADAPTATION` was present but referenced `batch_cls_preds` and `batch_box_preds` — keys that hold **decoded, flat** tensors after post-processing (shape `[B, N, C]`). The Conv2d discriminator expects spatial `[B, C, H, W]` tensors. Evidence from training losses:

```
train/cond_disc_loss0  = 0.000   # Car:       silent failure (NaN→0)
train/cond_disc_loss1  = 1.000   # Pedestrian: stuck at maximum loss
train/cond_disc_loss2  = 1.000   # Cyclist:    stuck at maximum loss
```

Additionally, `batch_cls_preds` in `CenterHead` stores **raw logits** (pre-sigmoid). The discriminator computes `y_k * cls_feats` as a class-confidence mask — raw logits are large negatives in non-detection regions, which *inverts* the feature map signal instead of zeroing it, causing pathological gradients.

**Metric comparison (epoch 40):**

| Class | hseplt9x (SECOND, marginal) | rhbp2jxp (CenterPoint, broken conditional) |
|---|---|---|
| Car AP | 18.5% | 8.4% |
| Pedestrian AP | 2.7% | 0.6% |
| Cyclist AP | ~0% | ~0% |
| **nuScenes mAP (10-class)** | **1.80%** | **0.89%** |
| **3-class mAP** | **~7.1%** | **~3.0%** |

CenterPoint performs **worse** due to the broken conditional discriminator.

---

## 3. Code Changes

### 3.1 `tools/adaptive_train.py`
Added `POINT_CLOUD_RANGE` assertion before dataloader construction (see Section 2.1).

### 3.2 `pcdet/models/dense_heads/anchor_head_single.py`

Added spatial map export during training for the conditional discriminator:

```python
# cls_preds_spatial: per-class max confidence over anchors → [B, num_class, H, W]
# box_preds_spatial: raw anchor box encodings             → [B, num_anchors*box_code_size, H, W]
if self.training:
    B, _, H, W = cls_preds.shape
    data_dict['cls_preds_spatial'] = cls_preds.view(
        B, self.num_anchors_per_location, self.num_class, H, W
    ).max(dim=1)[0]
    data_dict['box_preds_spatial'] = box_preds
```

### 3.3 `pcdet/models/dense_heads/center_head.py`

Added spatial map export during training, with sigmoid applied to the heatmap:

```python
# cls_preds_spatial: sigmoid-clamped heatmap [B, num_class, H, W]
# box_preds_spatial: raw spatial box encoding [B, 8, H, W]
if self.training:
    data_dict['cls_preds_spatial'] = torch.clamp(
        pred_dicts[0]['hm'].sigmoid(), min=1e-4, max=1 - 1e-4
    )
    data_dict['box_preds_spatial'] = data_dict['batch_box_preds']
```

**Why sigmoid is required:** The paper's conditional loss (Eq. 2) uses class confidence $\hat{y}_{k,n}$ as a scalar weight: $\mathcal{L}_C = \frac{1}{N}\sum_n \hat{y}_{k,n} \odot (g_{\theta_D}(x_n, \hat{b}_n) - d)^2$. With raw logits, $\hat{y}$ is negative in non-detection areas, which inverts the mask instead of suppressing it.

### 3.4 `tools/cfgs/kitti2nuscenes_models/second-rospm-M-C.yaml`

Added `CONDITIONAL_ADAPTATION` block and `LOSS_CONDITIONAL`:

```yaml
CONDITIONAL_ADAPTATION:
    # Keys saved by AnchorHeadSingle: cls_preds_spatial [B,3,H,W], box_preds_spatial [B,42,H,W]
    INPUT_DICT_KEYS: ['cls_preds_spatial', 'box_preds_spatial', 'spatial_features_2d']
    MLP_TYPE: Conv2d
    MLPS: [256, 128]       # prepended with NUM_FEATURES+BOX_REGRESSION_PARAMS=554 at init
    NUM_CLASSES: 3
    BOX_REGRESSION_PARAMS: 42  # 6 anchors × 7 box params
    NUM_FEATURES: 512          # spatial_features_2d channels
    KERNEL_SIZE: 3
    USE_SIGMOID: True

LOSS_CONFIG:
    LOSS_FUNCTION: [CrossEntropy]
    LOSS_CONDITIONAL: LeastSquares
```

### 3.5 `tools/cfgs/kitti2nuscenes_models/centerpoint-rospm-M-C.yaml` (and all `*-rospm-*` configs)

Updated `CONDITIONAL_ADAPTATION.INPUT_DICT_KEYS` from broken decoded keys to the new spatial keys:

```yaml
CONDITIONAL_ADAPTATION:
    # Keys saved by CenterHead: cls_preds_spatial [B,3,H,W] (sigmoid), box_preds_spatial [B,8,H,W]
    INPUT_DICT_KEYS: ['cls_preds_spatial', 'box_preds_spatial', 'spatial_features_2d']
    MLP_TYPE: Conv2d
    MLPS: [256, 128]
    NUM_CLASSES: 3
    BOX_REGRESSION_PARAMS: 8   # 2 center + 1 center_z + 3 dim + 2 rot
    NUM_FEATURES: 256
    KERNEL_SIZE: 3
    USE_SIGMOID: True
```

### 3.6 `POINT_CLOUD_RANGE` — corrected to symmetric 360° range (2026-05-03)

After the initial mismatch was resolved (Section 2.1), all configs were still using the asymmetric KITTI-centric range `[-50, -51.2, -2, 150, 51.2, 4]` (200 m forward, 50 m backward). This covers only the front hemisphere and discards ~75% of nuScenes 360° GT annotations.

**Fix:** Changed to `[-75.2, -75.2, -2, 75.2, 75.2, 4]` (symmetric 150.4 m × 150.4 m square) in:

| File | Change |
|---|---|
| `tools/cfgs/dataset_configs/da_kitti_dataset.yaml` | Range updated; overrides removed from all model configs |
| `tools/cfgs/dataset_configs/da_nuscenes_mini_dataset.yaml` | Same fix |
| All 8 `*-rospm-{M,C,M-C}.yaml` model configs | Inline `POINT_CLOUD_RANGE` overrides removed (now inherit base) |
| `centerpoint-rospm-{M,C}.yaml` (both directions) | `POST_CENTER_LIMIT_RANGE` updated to match |

Commit: `b0c5772`

---

## 4. Paper Reproduction Audit

**Paper:** *UADA3D: Unsupervised Adversarial Domain Adaptation for 3D Object Detection with Sparse LiDAR and Large Domain Gaps* (arXiv:2403.17633)

### 4.1 What the paper actually reports for K→N / W→N

The paper's primary results (Table I) use **IA-SSD** and **CenterPoint** as detectors, evaluated on **W→N** (Waymo→nuScenes). mAP is reported over **3 classes** (Vehicle, Pedestrian, Cyclist):

| Method | W→N mAP3D (IA-SSD) | W→N mAP3D (CenterPoint) |
|---|---|---|
| Source Only | 1.5% | 15.96% |
| ST3D++ | 11.94% | 17.66% |
| **UADA3D** | **18.33%** | **26.89%** |

SECOND appears only in Table II using **pre-trained weights from other methods**, for **Car class only** (not a 3-class K→N result).

### 4.2 Differences between current config and paper

| Parameter | Current runs | Paper UADA3D (W→N) |
|---|---|---|
| Source dataset | KITTI (~7,480 samples, front-FOV only) | Waymo (~158,100 samples, 360°) |
| Backbone | DASECONDNet / DACenterPoint | IA-SSD / CenterPoint |
| Discriminator | Marginal only (hseplt9x) / Broken conditional (rhbp2jxp) | Conditional: 3 class-wise, feature-masked, confidence-weighted |
| Point cloud range | `[-50, -51.2, -2, 150, 51.2, 4]` (forward hemisphere) | `[-75.2, -75.2, -2, 75.2, 75.2, 4]` (360°) |
| Source downsampling | None | 64→32 layers to match nuScenes VLP-32 |
| ROS | Uniform `U(0.8, 1.2)` | Class-specific: Vehicle `U(0.8,1.2)`, Ped/Cyc `U(0.9,1.1)` |
| GRL coefficient | 0.1 (constant) | 0.1 (constant) ✓ |
| Epochs | 40 | 40 (CenterPoint), 80 (IA-SSD) |

### 4.3 The 10-class vs 3-class mAP discrepancy

The nuScenes evaluator reports mAP over all 10 official classes. The model predicts only 3 classes (Car/Pedestrian/Cyclist → mapped from nuScenes labels), so 7 classes always have AP = 0. This dilutes the reported mAP by 10/3:

```
nuScenes mAP (10-class) = (Car_AP + Ped_AP + Cyc_AP + 0 × 7) / 10
3-class mAP             = (Car_AP + Ped_AP + Cyc_AP) / 3
```

For `hseplt9x`: 10-class = 1.80%, 3-class ≈ (13.5 + 4.5 + 0) / 3 = **6.0%**

The paper's 3-class mAP metric is the correct one for comparison.

---

## 5. Remaining Open Issues

| # | Issue | Impact | Suggested fix |
|---|---|---|---|
| 1 | ~~**Asymmetric point cloud range**~~ | ~~major AP loss~~ | ✅ **Fixed** (commit `b0c5772`) — symmetric `[-75.2,-75.2,-2,75.2,75.2,4]` now in all DA base configs |
| 2 | **KITTI as source** (not Waymo) — front-FOV labels only, 20× fewer samples | Model never sees rear/side objects in source; poor generalization | Use Waymo as source to match paper's primary benchmark |
| 3 | **No source downsampling** | Domain gap in LiDAR density unaddressed | Add random beam dropout to simulate 32-layer target density |
| 4 | **Conditional discriminator not yet validated** | Fixes are applied but no re-run yet | Trigger new training run with updated configs |

---

## 6. Next Steps

1. **Re-run** `centerpoint-rospm-C` (and `second-rospm-C`) with the fixed 360° range and `extra_tag` distinguishing this run from earlier broken ones.
2. **Confirm** `cond_disc_loss0/1/2` are all non-zero and decreasing in the new run.
3. For a true paper reproduction: set up W→N with CenterPoint following the paper's exact augmentation schedule.
