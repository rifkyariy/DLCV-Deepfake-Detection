# Inference Report — IMG_5547

## Overview
- **Process ID:** fsbi_inference_IMG_5547_20251201_190225
- **Timestamp:** 2025-12-01T19:02:35.376466
- **Device:** cuda
- **Weights:** /app/weights/best_model.tar
- **Input Image:** /app/input/test/IMG_5547.jpg
- **Grad-CAM target:** fake (index 1)

## Step A — Input Image
- Dimensions: 824x649 (HxW)

![Original input](steps/a_input/original.png)

![Detection overlay](steps/a_input/retinaface_overlay.png)

## Step B — Face Detection

1. **Source/Target Generator (SBI):** During training, the base frame is duplicated into target and source views. Augmentations applied to either branch mimic compression, color, and motion drift so the model sees realistic perturbations.
2. **Mask Generator:** Landmarks and RetinaFace detections create a convex-hull face mask (or `random_get_hull` when available). Additional affine + elastic warps make every mask unique.
3. **Blending (Self-Blended Image):** The source face is composited onto the target face through the generated mask using `dynamic_blend`, yielding an SBI pair (fake vs real) that the detector learns from.

### Runtime Face Detection Snapshot
- Bounding box (x0, y0, x1, y1): [276.0, 106.0, 569.0, 522.0]
- Landmarks (x, y): [[354.0, 288.0], [490.0, 255.0], [439.0, 360.0], [396.0, 434.0], [504.0, 410.0]]

![Cropped face](steps/b_preprocess/cropped_face.png)

### Landmark Mask Approximation
![Mask only](steps/b_preprocess/mask.png)

## Step C — Frequency Features (DWT/FFG)

### Channel-Wise Wavelet Pass (sym2, reflect)
a. **DWT from R:** Extract approximation map `cA_r`, resize, blend with original red channel `(cA_r + R) / 2`.
b. **DWT from G:** Same pipeline for the green channel producing `cA_g`.
c. **DWT from B:** Blue channel variant yielding `cA_b`.
d. **Combine as Channels:** Stack `[cA_r, cA_g, cA_b]` to form the FSBI tensor fed to EfficientNet-B5.

### Visual Comparison
![Original face](steps/b_preprocess/cropped_face.png)

![Combined channels](steps/b_preprocess/dwt_processed.png)

### Channel Breakdown
Channel | Visualization
---|---
Red (cA_r) | ![](steps/b_preprocess/dwt_red.png)
Green (cA_g) | ![](steps/b_preprocess/dwt_green.png)
Blue (cA_b) | ![](steps/b_preprocess/dwt_blue.png)

### Statistics
- **Blended tensor:** mean 0.2479, std 0.2437, min -0.0013, max 1.4831
- **cA_r:** mean 0.4191, std 0.3499
- **cA_g:** mean 0.3286, std 0.3308
- **cA_b:** mean 0.2454, std 0.2665

## Step D — Feature Journey
Layer | Shape | Activation | Heatmap
---|---|---|---
conv_stem | (1, 48, 190, 190) | ![](steps/c_layers/conv_stem/image/activation.png) | ![](steps/c_layers/conv_stem/heatmap/overlay.png)
bn0 | (1, 48, 190, 190) | ![](steps/c_layers/bn0/image/activation.png) | ![](steps/c_layers/bn0/heatmap/overlay.png)
block_00 | (1, 24, 190, 190) | ![](steps/c_layers/block_00/image/activation.png) | ![](steps/c_layers/block_00/heatmap/overlay.png)
block_01 | (1, 24, 190, 190) | ![](steps/c_layers/block_01/image/activation.png) | ![](steps/c_layers/block_01/heatmap/overlay.png)
block_02 | (1, 24, 190, 190) | ![](steps/c_layers/block_02/image/activation.png) | ![](steps/c_layers/block_02/heatmap/overlay.png)
block_03 | (1, 40, 95, 95) | ![](steps/c_layers/block_03/image/activation.png) | ![](steps/c_layers/block_03/heatmap/overlay.png)
block_04 | (1, 40, 95, 95) | ![](steps/c_layers/block_04/image/activation.png) | ![](steps/c_layers/block_04/heatmap/overlay.png)
block_05 | (1, 40, 95, 95) | ![](steps/c_layers/block_05/image/activation.png) | ![](steps/c_layers/block_05/heatmap/overlay.png)
block_06 | (1, 40, 95, 95) | ![](steps/c_layers/block_06/image/activation.png) | ![](steps/c_layers/block_06/heatmap/overlay.png)
block_07 | (1, 40, 95, 95) | ![](steps/c_layers/block_07/image/activation.png) | ![](steps/c_layers/block_07/heatmap/overlay.png)
block_08 | (1, 64, 47, 47) | ![](steps/c_layers/block_08/image/activation.png) | ![](steps/c_layers/block_08/heatmap/overlay.png)
block_09 | (1, 64, 47, 47) | ![](steps/c_layers/block_09/image/activation.png) | ![](steps/c_layers/block_09/heatmap/overlay.png)
block_10 | (1, 64, 47, 47) | ![](steps/c_layers/block_10/image/activation.png) | ![](steps/c_layers/block_10/heatmap/overlay.png)
block_11 | (1, 64, 47, 47) | ![](steps/c_layers/block_11/image/activation.png) | ![](steps/c_layers/block_11/heatmap/overlay.png)
block_12 | (1, 64, 47, 47) | ![](steps/c_layers/block_12/image/activation.png) | ![](steps/c_layers/block_12/heatmap/overlay.png)
block_13 | (1, 128, 24, 24) | ![](steps/c_layers/block_13/image/activation.png) | ![](steps/c_layers/block_13/heatmap/overlay.png)
block_14 | (1, 128, 24, 24) | ![](steps/c_layers/block_14/image/activation.png) | ![](steps/c_layers/block_14/heatmap/overlay.png)
block_15 | (1, 128, 24, 24) | ![](steps/c_layers/block_15/image/activation.png) | ![](steps/c_layers/block_15/heatmap/overlay.png)
block_16 | (1, 128, 24, 24) | ![](steps/c_layers/block_16/image/activation.png) | ![](steps/c_layers/block_16/heatmap/overlay.png)
block_17 | (1, 128, 24, 24) | ![](steps/c_layers/block_17/image/activation.png) | ![](steps/c_layers/block_17/heatmap/overlay.png)
block_18 | (1, 128, 24, 24) | ![](steps/c_layers/block_18/image/activation.png) | ![](steps/c_layers/block_18/heatmap/overlay.png)
block_19 | (1, 128, 24, 24) | ![](steps/c_layers/block_19/image/activation.png) | ![](steps/c_layers/block_19/heatmap/overlay.png)
block_20 | (1, 176, 24, 24) | ![](steps/c_layers/block_20/image/activation.png) | ![](steps/c_layers/block_20/heatmap/overlay.png)
block_21 | (1, 176, 24, 24) | ![](steps/c_layers/block_21/image/activation.png) | ![](steps/c_layers/block_21/heatmap/overlay.png)
block_22 | (1, 176, 24, 24) | ![](steps/c_layers/block_22/image/activation.png) | ![](steps/c_layers/block_22/heatmap/overlay.png)
block_23 | (1, 176, 24, 24) | ![](steps/c_layers/block_23/image/activation.png) | ![](steps/c_layers/block_23/heatmap/overlay.png)
block_24 | (1, 176, 24, 24) | ![](steps/c_layers/block_24/image/activation.png) | ![](steps/c_layers/block_24/heatmap/overlay.png)
block_25 | (1, 176, 24, 24) | ![](steps/c_layers/block_25/image/activation.png) | ![](steps/c_layers/block_25/heatmap/overlay.png)
block_26 | (1, 176, 24, 24) | ![](steps/c_layers/block_26/image/activation.png) | ![](steps/c_layers/block_26/heatmap/overlay.png)
block_27 | (1, 304, 12, 12) | ![](steps/c_layers/block_27/image/activation.png) | ![](steps/c_layers/block_27/heatmap/overlay.png)
block_28 | (1, 304, 12, 12) | ![](steps/c_layers/block_28/image/activation.png) | ![](steps/c_layers/block_28/heatmap/overlay.png)
block_29 | (1, 304, 12, 12) | ![](steps/c_layers/block_29/image/activation.png) | ![](steps/c_layers/block_29/heatmap/overlay.png)
block_30 | (1, 304, 12, 12) | ![](steps/c_layers/block_30/image/activation.png) | ![](steps/c_layers/block_30/heatmap/overlay.png)
block_31 | (1, 304, 12, 12) | ![](steps/c_layers/block_31/image/activation.png) | ![](steps/c_layers/block_31/heatmap/overlay.png)
block_32 | (1, 304, 12, 12) | ![](steps/c_layers/block_32/image/activation.png) | ![](steps/c_layers/block_32/heatmap/overlay.png)
block_33 | (1, 304, 12, 12) | ![](steps/c_layers/block_33/image/activation.png) | ![](steps/c_layers/block_33/heatmap/overlay.png)
block_34 | (1, 304, 12, 12) | ![](steps/c_layers/block_34/image/activation.png) | ![](steps/c_layers/block_34/heatmap/overlay.png)
block_35 | (1, 304, 12, 12) | ![](steps/c_layers/block_35/image/activation.png) | ![](steps/c_layers/block_35/heatmap/overlay.png)
block_36 | (1, 512, 12, 12) | ![](steps/c_layers/block_36/image/activation.png) | ![](steps/c_layers/block_36/heatmap/overlay.png)
block_37 | (1, 512, 12, 12) | ![](steps/c_layers/block_37/image/activation.png) | ![](steps/c_layers/block_37/heatmap/overlay.png)
block_38 | (1, 512, 12, 12) | ![](steps/c_layers/block_38/image/activation.png) | ![](steps/c_layers/block_38/heatmap/overlay.png)
conv_head | (1, 2048, 12, 12) | ![](steps/c_layers/conv_head/image/activation.png) | ![](steps/c_layers/conv_head/heatmap/overlay.png)
bn1 | (1, 2048, 12, 12) | ![](steps/c_layers/bn1/image/activation.png) | ![](steps/c_layers/bn1/heatmap/overlay.png)
avg_pool | (1, 2048, 1, 1) | ![](steps/c_layers/avg_pool/image/activation.png) | ![](steps/c_layers/avg_pool/heatmap/overlay.png)
dropout | (1, 2048) | ![](steps/c_layers/dropout/image/activation.png) | ![](steps/c_layers/dropout/heatmap/overlay.png)
fc | (1, 2) | ![](steps/c_layers/fc/image/activation.png) | ![](steps/c_layers/fc/heatmap/overlay.png)

## Step E — Classifier
- Logits: [-0.887061357498169, 0.9705662131309509]
- Probabilities: [0.13497981429100037, 0.865020215511322]
- Predicted label: **fake**
- Class confidence: 0.8650