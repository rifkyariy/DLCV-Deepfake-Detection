# Deepfake Analysis Report (Backbone: b5)
---
## 1. Performance Summary
| Model | Size (MB) | Reduction | AUC | Diff |
| :--- | :--- | :--- | :--- | :--- |
| **baseline** | 108.79 | -0.0% | 0.9549 (Baseline) | +0.0000 |
| **q16** | 54.60 | -49.8% | 0.9548 (Stable) | -0.0001 |
| **q16-p50** | 54.60 | -49.8% | 0.9334 (Degraded) | -0.0215 |
| **q32-p50** | 108.99 | --0.2% | 0.9332 (Degraded) | -0.0216 |
| **q8** | 27.80 | -74.4% | 0.7588 (Degraded) | -0.1960 |
| **q8-p50** | 27.80 | -74.4% | 0.7119 (Degraded) | -0.2429 |

## 2. Key Insights
- Highest AUC: **baseline** at 0.9549.
- Best compression within 1% AUC of baseline: **q16** (54.6 MB, 49.8% smaller).
- Largest AUC drop: **q8-p50** (-0.2429).

## 3. Classification Quality
| Model | Samples | TP | TN | FP | FN | Precision | Recall | F1 | Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **baseline** | 518 | 292 | 163 | 15 | 48 | 0.951 | 0.859 | 0.903 | 0.878 |
| **q16** | 518 | 291 | 163 | 15 | 49 | 0.951 | 0.856 | 0.901 | 0.876 |
| **q16-p50** | 518 | 339 | 50 | 128 | 1 | 0.726 | 0.997 | 0.840 | 0.751 |
| **q32-p50** | 518 | 339 | 49 | 129 | 1 | 0.724 | 0.997 | 0.839 | 0.749 |
| **q8** | 518 | 339 | 2 | 176 | 1 | 0.658 | 0.997 | 0.793 | 0.658 |
| **q8-p50** | 518 | 340 | 0 | 178 | 0 | 0.656 | 1.000 | 0.793 | 0.656 |

## 4. Visualizations
| Size vs AUC | ROC Curve |
| :---: | :---: |
| ![Bar](performance_graph.png) | ![ROC](roc_curve_comparison.png) |

## 5. Visual Analysis
### Resolved Fakes (True Positives)

| Model | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | ![img](../results/baseline/result/true_positives/heatmap/id37_id3_0004.mp4.jpg)<br>`0.9999`<br>_id37_id3_0.._ | ![img](../results/baseline/result/true_positives/heatmap/id46_id41_0000.mp4.jpg)<br>`0.9999`<br>_id46_id41_.._ | ![img](../results/baseline/result/true_positives/heatmap/id43_id40_0005.mp4.jpg)<br>`0.9999`<br>_id43_id40_.._ | ![img](../results/baseline/result/true_positives/heatmap/id1_id2_0007.mp4.jpg)<br>`0.9999`<br>_id1_id2_00.._ | ![img](../results/baseline/result/true_positives/heatmap/id17_id16_0000.mp4.jpg)<br>`0.9998`<br>_id17_id16_.._ |
| **q16** | ![img](../results/q16/result/true_positives/heatmap/id37_id3_0004.mp4.jpg)<br>`0.9999`<br>_id37_id3_0.._ | ![img](../results/q16/result/true_positives/heatmap/id46_id41_0000.mp4.jpg)<br>`0.9999`<br>_id46_id41_.._ | ![img](../results/q16/result/true_positives/heatmap/id43_id40_0005.mp4.jpg)<br>`0.9999`<br>_id43_id40_.._ | ![img](../results/q16/result/true_positives/heatmap/id1_id2_0007.mp4.jpg)<br>`0.9999`<br>_id1_id2_00.._ | ![img](../results/q16/result/true_positives/heatmap/id17_id16_0000.mp4.jpg)<br>`0.9998`<br>_id17_id16_.._ |
| **q16-p50** | ![img](../results/q16-p50/result/true_positives/heatmap/id35_id31_0006.mp4.jpg)<br>`0.9986`<br>_id35_id31_.._ | ![img](../results/q16-p50/result/true_positives/heatmap/id37_id28_0007.mp4.jpg)<br>`0.9982`<br>_id37_id28_.._ | ![img](../results/q16-p50/result/true_positives/heatmap/id21_id20_0006.mp4.jpg)<br>`0.9981`<br>_id21_id20_.._ | ![img](../results/q16-p50/result/true_positives/heatmap/id39_id44_0008.mp4.jpg)<br>`0.9980`<br>_id39_id44_.._ | ![img](../results/q16-p50/result/true_positives/heatmap/id28_id6_0006.mp4.jpg)<br>`0.9980`<br>_id28_id6_0.._ |
| **q32-p50** | ![img](../results/q32-p50/result/true_positives/heatmap/id35_id31_0006.mp4.jpg)<br>`0.9986`<br>_id35_id31_.._ | ![img](../results/q32-p50/result/true_positives/heatmap/id37_id28_0007.mp4.jpg)<br>`0.9982`<br>_id37_id28_.._ | ![img](../results/q32-p50/result/true_positives/heatmap/id21_id20_0006.mp4.jpg)<br>`0.9981`<br>_id21_id20_.._ | ![img](../results/q32-p50/result/true_positives/heatmap/id39_id44_0008.mp4.jpg)<br>`0.9981`<br>_id39_id44_.._ | ![img](../results/q32-p50/result/true_positives/heatmap/id28_id6_0006.mp4.jpg)<br>`0.9980`<br>_id28_id6_0.._ |
| **q8** | ![img](../results/q8/result/true_positives/heatmap/id46_id41_0000.mp4.jpg)<br>`1.0000`<br>_id46_id41_.._ | ![img](../results/q8/result/true_positives/heatmap/id2_id0_0008.mp4.jpg)<br>`1.0000`<br>_id2_id0_00.._ | ![img](../results/q8/result/true_positives/heatmap/id1_id2_0002.mp4.jpg)<br>`1.0000`<br>_id1_id2_00.._ | ![img](../results/q8/result/true_positives/heatmap/id1_id3_0003.mp4.jpg)<br>`0.9999`<br>_id1_id3_00.._ | ![img](../results/q8/result/true_positives/heatmap/id57_id53_0006.mp4.jpg)<br>`0.9999`<br>_id57_id53_.._ |
| **q8-p50** | ![img](../results/q8-p50/result/true_positives/heatmap/id1_id2_0007.mp4.jpg)<br>`0.9998`<br>_id1_id2_00.._ | ![img](../results/q8-p50/result/true_positives/heatmap/id17_id2_0000.mp4.jpg)<br>`0.9995`<br>_id17_id2_0.._ | ![img](../results/q8-p50/result/true_positives/heatmap/id2_id26_0001.mp4.jpg)<br>`0.9994`<br>_id2_id26_0.._ | ![img](../results/q8-p50/result/true_positives/heatmap/id24_id20_0009.mp4.jpg)<br>`0.9993`<br>_id24_id20_.._ | ![img](../results/q8-p50/result/true_positives/heatmap/id2_id0_0008.mp4.jpg)<br>`0.9993`<br>_id2_id0_00.._ |


### Missed Fakes (False Negatives)

| Model | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | ![img](../results/baseline/result/false_negatives/heatmap/id29_id32_0000.mp4.jpg)<br>`0.0304`<br>_id29_id32_.._ | ![img](../results/baseline/result/false_negatives/heatmap/id30_id6_0007.mp4.jpg)<br>`0.1034`<br>_id30_id6_0.._ | ![img](../results/baseline/result/false_negatives/heatmap/id33_id32_0006.mp4.jpg)<br>`0.1215`<br>_id33_id32_.._ | ![img](../results/baseline/result/false_negatives/heatmap/id30_id23_0007.mp4.jpg)<br>`0.1218`<br>_id30_id23_.._ | ![img](../results/baseline/result/false_negatives/heatmap/id9_id3_0009.mp4.jpg)<br>`0.1294`<br>_id9_id3_00.._ |
| **q16** | ![img](../results/q16/result/false_negatives/heatmap/id29_id32_0000.mp4.jpg)<br>`0.0301`<br>_id29_id32_.._ | ![img](../results/q16/result/false_negatives/heatmap/id30_id6_0007.mp4.jpg)<br>`0.1025`<br>_id30_id6_0.._ | ![img](../results/q16/result/false_negatives/heatmap/id33_id32_0006.mp4.jpg)<br>`0.1206`<br>_id33_id32_.._ | ![img](../results/q16/result/false_negatives/heatmap/id30_id23_0007.mp4.jpg)<br>`0.1208`<br>_id30_id23_.._ | ![img](../results/q16/result/false_negatives/heatmap/id9_id3_0009.mp4.jpg)<br>`0.1291`<br>_id9_id3_00.._ |
| **q16-p50** | ![img](../results/q16-p50/result/false_negatives/heatmap/id9_id3_0009.mp4.jpg)<br>`0.3961`<br>_id9_id3_00.._ | N/A | N/A | N/A | N/A |
| **q32-p50** | ![img](../results/q32-p50/result/false_negatives/heatmap/id9_id3_0009.mp4.jpg)<br>`0.3973`<br>_id9_id3_00.._ | N/A | N/A | N/A | N/A |
| **q8** | ![img](../results/q8/result/false_negatives/heatmap/id10_id11_0001.mp4.jpg)<br>`0.3872`<br>_id10_id11_.._ | N/A | N/A | N/A | N/A |
| **q8-p50** | N/A | N/A | N/A | N/A | N/A |

### False Positives (Real labeled Fake)

| Model | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | ![img](../results/baseline/result/false_positives/heatmap/00047.mp4.jpg)<br>`0.9808`<br>_00047.mp4_ | ![img](../results/baseline/result/false_positives/heatmap/id51_0001.mp4.jpg)<br>`0.8435`<br>_id51_0001..._ | ![img](../results/baseline/result/false_positives/heatmap/00170.mp4.jpg)<br>`0.8011`<br>_00170.mp4_ | ![img](../results/baseline/result/false_positives/heatmap/id31_0003.mp4.jpg)<br>`0.7185`<br>_id31_0003..._ | ![img](../results/baseline/result/false_positives/heatmap/id21_0009.mp4.jpg)<br>`0.6559`<br>_id21_0009..._ |
| **q16** | ![img](../results/q16/result/false_positives/heatmap/00047.mp4.jpg)<br>`0.9809`<br>_00047.mp4_ | ![img](../results/q16/result/false_positives/heatmap/id51_0001.mp4.jpg)<br>`0.8418`<br>_id51_0001..._ | ![img](../results/q16/result/false_positives/heatmap/00170.mp4.jpg)<br>`0.8003`<br>_00170.mp4_ | ![img](../results/q16/result/false_positives/heatmap/id31_0003.mp4.jpg)<br>`0.7177`<br>_id31_0003..._ | ![img](../results/q16/result/false_positives/heatmap/id21_0009.mp4.jpg)<br>`0.6526`<br>_id21_0009..._ |
| **q16-p50** | ![img](../results/q16-p50/result/false_positives/heatmap/00047.mp4.jpg)<br>`0.9966`<br>_00047.mp4_ | ![img](../results/q16-p50/result/false_positives/heatmap/id51_0001.mp4.jpg)<br>`0.9955`<br>_id51_0001..._ | ![img](../results/q16-p50/result/false_positives/heatmap/00082.mp4.jpg)<br>`0.9936`<br>_00082.mp4_ | ![img](../results/q16-p50/result/false_positives/heatmap/00256.mp4.jpg)<br>`0.9876`<br>_00256.mp4_ | ![img](../results/q16-p50/result/false_positives/heatmap/00264.mp4.jpg)<br>`0.9728`<br>_00264.mp4_ |
| **q32-p50** | ![img](../results/q32-p50/result/false_positives/heatmap/00047.mp4.jpg)<br>`0.9966`<br>_00047.mp4_ | ![img](../results/q32-p50/result/false_positives/heatmap/id51_0001.mp4.jpg)<br>`0.9955`<br>_id51_0001..._ | ![img](../results/q32-p50/result/false_positives/heatmap/00082.mp4.jpg)<br>`0.9937`<br>_00082.mp4_ | ![img](../results/q32-p50/result/false_positives/heatmap/00256.mp4.jpg)<br>`0.9878`<br>_00256.mp4_ | ![img](../results/q32-p50/result/false_positives/heatmap/00264.mp4.jpg)<br>`0.9734`<br>_00264.mp4_ |
| **q8** | ![img](../results/q8/result/false_positives/heatmap/00119.mp4.jpg)<br>`0.9999`<br>_00119.mp4_ | ![img](../results/q8/result/false_positives/heatmap/00236.mp4.jpg)<br>`0.9999`<br>_00236.mp4_ | ![img](../results/q8/result/false_positives/heatmap/id37_0004.mp4.jpg)<br>`0.9999`<br>_id37_0004..._ | ![img](../results/q8/result/false_positives/heatmap/00252.mp4.jpg)<br>`0.9998`<br>_00252.mp4_ | ![img](../results/q8/result/false_positives/heatmap/00264.mp4.jpg)<br>`0.9998`<br>_00264.mp4_ |
| **q8-p50** | ![img](../results/q8-p50/result/false_positives/heatmap/00264.mp4.jpg)<br>`0.9993`<br>_00264.mp4_ | ![img](../results/q8-p50/result/false_positives/heatmap/00252.mp4.jpg)<br>`0.9989`<br>_00252.mp4_ | ![img](../results/q8-p50/result/false_positives/heatmap/id13_0011.mp4.jpg)<br>`0.9987`<br>_id13_0011..._ | ![img](../results/q8-p50/result/false_positives/heatmap/00256.mp4.jpg)<br>`0.9984`<br>_00256.mp4_ | ![img](../results/q8-p50/result/false_positives/heatmap/00119.mp4.jpg)<br>`0.9984`<br>_00119.mp4_ |


### Confirmed Reals (True Negatives)

| Model | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | ![img](../results/baseline/result/true_negatives/heatmap/id32_0008.mp4.jpg)<br>`0.0020`<br>_id32_0008..._ | ![img](../results/baseline/result/true_negatives/heatmap/id27_0009.mp4.jpg)<br>`0.0026`<br>_id27_0009..._ | ![img](../results/baseline/result/true_negatives/heatmap/id4_0004.mp4.jpg)<br>`0.0027`<br>_id4_0004.mp4_ | ![img](../results/baseline/result/true_negatives/heatmap/id36_0008.mp4.jpg)<br>`0.0033`<br>_id36_0008..._ | ![img](../results/baseline/result/true_negatives/heatmap/id44_0003.mp4.jpg)<br>`0.0035`<br>_id44_0003..._ |
| **q16** | ![img](../results/q16/result/true_negatives/heatmap/id32_0008.mp4.jpg)<br>`0.0020`<br>_id32_0008..._ | ![img](../results/q16/result/true_negatives/heatmap/id27_0009.mp4.jpg)<br>`0.0026`<br>_id27_0009..._ | ![img](../results/q16/result/true_negatives/heatmap/id4_0004.mp4.jpg)<br>`0.0027`<br>_id4_0004.mp4_ | ![img](../results/q16/result/true_negatives/heatmap/id36_0008.mp4.jpg)<br>`0.0033`<br>_id36_0008..._ | ![img](../results/q16/result/true_negatives/heatmap/id44_0003.mp4.jpg)<br>`0.0035`<br>_id44_0003..._ |
| **q16-p50** | ![img](../results/q16-p50/result/true_negatives/heatmap/id32_0008.mp4.jpg)<br>`0.0321`<br>_id32_0008..._ | ![img](../results/q16-p50/result/true_negatives/heatmap/id36_0008.mp4.jpg)<br>`0.0702`<br>_id36_0008..._ | ![img](../results/q16-p50/result/true_negatives/heatmap/id27_0009.mp4.jpg)<br>`0.0847`<br>_id27_0009..._ | ![img](../results/q16-p50/result/true_negatives/heatmap/id35_0003.mp4.jpg)<br>`0.0956`<br>_id35_0003..._ | ![img](../results/q16-p50/result/true_negatives/heatmap/id36_0000.mp4.jpg)<br>`0.1100`<br>_id36_0000..._ |
| **q32-p50** | ![img](../results/q32-p50/result/true_negatives/heatmap/id32_0008.mp4.jpg)<br>`0.0323`<br>_id32_0008..._ | ![img](../results/q32-p50/result/true_negatives/heatmap/id36_0008.mp4.jpg)<br>`0.0704`<br>_id36_0008..._ | ![img](../results/q32-p50/result/true_negatives/heatmap/id27_0009.mp4.jpg)<br>`0.0851`<br>_id27_0009..._ | ![img](../results/q32-p50/result/true_negatives/heatmap/id35_0003.mp4.jpg)<br>`0.0965`<br>_id35_0003..._ | ![img](../results/q32-p50/result/true_negatives/heatmap/id36_0000.mp4.jpg)<br>`0.1107`<br>_id36_0000..._ |
| **q8** | ![img](../results/q8/result/true_negatives/heatmap/00207.mp4.jpg)<br>`0.3014`<br>_00207.mp4_ | ![img](../results/q8/result/true_negatives/heatmap/id35_0003.mp4.jpg)<br>`0.3667`<br>_id35_0003..._ | N/A | N/A | N/A |
| **q8-p50** | N/A | N/A | N/A | N/A | N/A |
