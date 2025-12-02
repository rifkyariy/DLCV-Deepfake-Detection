# Deepfake Model Analysis Report
**Backbone:** EfficientNet-b5 | **Base Size:** 109.07 MB
---
## 1. Performance Summary
| Model | Size (MB) | Reduction | AUC | Diff |
| :--- | :--- | :--- | :--- | :--- |
| **baseline** | 109.07 | -0.0% | 0.9549 ⚪ | +0.0000 |
| **Comp-p50** | 108.99 | -0.1% | 0.9332 🔴 | -0.0216 |
| **Comp-q16-p0** | 54.60 | -49.9% | 0.9548 🟢 | -0.0001 |
| **Comp-q16-p50** | 54.60 | -49.9% | 0.9334 🔴 | -0.0215 |
| **Comp-q8** | 27.80 | -74.5% | 0.7588 🔴 | -0.1960 |
| **Comp-q8-p50** | 27.80 | -74.5% | 0.7119 🔴 | -0.2429 |


## 2. Visualizations
| Size vs AUC | ROC Curve |
| :---: | :---: |
| ![Bar](performance_graph.png) | ![ROC](roc_curve_comparison.png) |


## 3. Visual Analysis (Heatmaps)
Comparison of model attention on resolved (correct) and unresolved (missed) fakes.
### Resolved Heatmaps (Top 5)

| Model | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | ![img](../results/baseline/result/resolved/heatmap/rank1_id37_id3_0004.mp4.png)<br>`0.9999`<br>_id37_id3_000.._ | ![img](../results/baseline/result/resolved/heatmap/rank2_id46_id41_0000.mp4.png)<br>`0.9999`<br>_id46_id41_00.._ | ![img](../results/baseline/result/resolved/heatmap/rank3_id43_id40_0005.mp4.png)<br>`0.9999`<br>_id43_id40_00.._ | ![img](../results/baseline/result/resolved/heatmap/rank4_id1_id2_0007.mp4.png)<br>`0.9999`<br>_id1_id2_0007.._ | ![img](../results/baseline/result/resolved/heatmap/rank5_id17_id16_0000.mp4.png)<br>`0.9998`<br>_id17_id16_00.._ |
| **Comp-p50** | ![img](../results/p50/result/resolved/heatmap/rank1_id35_id31_0006.mp4.png)<br>`0.9986`<br>_id35_id31_00.._ | ![img](../results/p50/result/resolved/heatmap/rank2_id37_id28_0007.mp4.png)<br>`0.9982`<br>_id37_id28_00.._ | ![img](../results/p50/result/resolved/heatmap/rank3_id21_id20_0006.mp4.png)<br>`0.9981`<br>_id21_id20_00.._ | ![img](../results/p50/result/resolved/heatmap/rank4_id39_id44_0008.mp4.png)<br>`0.9981`<br>_id39_id44_00.._ | ![img](../results/p50/result/resolved/heatmap/rank5_id28_id6_0006.mp4.png)<br>`0.9980`<br>_id28_id6_000.._ |
| **Comp-q16-p0** | ![img](../results/q16-p0/result/resolved/heatmap/rank1_id37_id3_0004.mp4.png)<br>`0.9999`<br>_id37_id3_000.._ | ![img](../results/q16-p0/result/resolved/heatmap/rank2_id46_id41_0000.mp4.png)<br>`0.9999`<br>_id46_id41_00.._ | ![img](../results/q16-p0/result/resolved/heatmap/rank3_id43_id40_0005.mp4.png)<br>`0.9999`<br>_id43_id40_00.._ | ![img](../results/q16-p0/result/resolved/heatmap/rank4_id1_id2_0007.mp4.png)<br>`0.9999`<br>_id1_id2_0007.._ | ![img](../results/q16-p0/result/resolved/heatmap/rank5_id17_id16_0000.mp4.png)<br>`0.9998`<br>_id17_id16_00.._ |
| **Comp-q16-p50** | ![img](../results/q16-p50/result/resolved/heatmap/rank1_id35_id31_0006.mp4.png)<br>`0.9986`<br>_id35_id31_00.._ | ![img](../results/q16-p50/result/resolved/heatmap/rank2_id37_id28_0007.mp4.png)<br>`0.9982`<br>_id37_id28_00.._ | ![img](../results/q16-p50/result/resolved/heatmap/rank3_id21_id20_0006.mp4.png)<br>`0.9981`<br>_id21_id20_00.._ | ![img](../results/q16-p50/result/resolved/heatmap/rank4_id39_id44_0008.mp4.png)<br>`0.9980`<br>_id39_id44_00.._ | ![img](../results/q16-p50/result/resolved/heatmap/rank5_id28_id6_0006.mp4.png)<br>`0.9980`<br>_id28_id6_000.._ |
| **Comp-q8** | ![img](../results/q8/result/resolved/heatmap/rank1_id46_id41_0000.mp4.png)<br>`1.0000`<br>_id46_id41_00.._ | ![img](../results/q8/result/resolved/heatmap/rank2_id2_id0_0008.mp4.png)<br>`1.0000`<br>_id2_id0_0008.._ | ![img](../results/q8/result/resolved/heatmap/rank3_id1_id2_0002.mp4.png)<br>`1.0000`<br>_id1_id2_0002.._ | ![img](../results/q8/result/resolved/heatmap/rank4_id1_id3_0003.mp4.png)<br>`0.9999`<br>_id1_id3_0003.._ | ![img](../results/q8/result/resolved/heatmap/rank5_id57_id53_0006.mp4.png)<br>`0.9999`<br>_id57_id53_00.._ |
| **Comp-q8-p50** | ![img](../results/q8-p50/result/resolved/heatmap/rank1_id1_id2_0007.mp4.png)<br>`0.9998`<br>_id1_id2_0007.._ | ![img](../results/q8-p50/result/resolved/heatmap/rank2_id17_id2_0000.mp4.png)<br>`0.9995`<br>_id17_id2_000.._ | ![img](../results/q8-p50/result/resolved/heatmap/rank3_id2_id26_0001.mp4.png)<br>`0.9994`<br>_id2_id26_000.._ | ![img](../results/q8-p50/result/resolved/heatmap/rank4_id24_id20_0009.mp4.png)<br>`0.9993`<br>_id24_id20_00.._ | ![img](../results/q8-p50/result/resolved/heatmap/rank5_id2_id0_0008.mp4.png)<br>`0.9993`<br>_id2_id0_0008.._ |

<br>
### Unresolved Heatmaps (Top 5)

| Model | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | ![img](../results/baseline/result/unresolved/heatmap/rank1_id21_id19_0005.mp4.png)<br>`0.2413`<br>_id21_id19_00.._ | ![img](../results/baseline/result/unresolved/heatmap/rank2_id38_id34_0004.mp4.png)<br>`0.2252`<br>_id38_id34_00.._ | ![img](../results/baseline/result/unresolved/heatmap/rank3_id4_id6_0008.mp4.png)<br>`0.2219`<br>_id4_id6_0008.._ | ![img](../results/baseline/result/unresolved/heatmap/rank4_id34_id32_0007.mp4.png)<br>`0.2019`<br>_id34_id32_00.._ | ![img](../results/baseline/result/unresolved/heatmap/rank5_id13_id7_0000.mp4.png)<br>`0.1330`<br>_id13_id7_000.._ |
| **Comp-p50** | ![img](../results/p50/result/unresolved/heatmap/rank1_id31_id16_0002.mp4.png)<br>`0.7619`<br>_id31_id16_00.._ | ![img](../results/p50/result/unresolved/heatmap/rank2_id34_id32_0007.mp4.png)<br>`0.7375`<br>_id34_id32_00.._ | ![img](../results/p50/result/unresolved/heatmap/rank3_id30_id23_0007.mp4.png)<br>`0.6920`<br>_id30_id23_00.._ | ![img](../results/p50/result/unresolved/heatmap/rank4_id30_id6_0007.mp4.png)<br>`0.6804`<br>_id30_id6_000.._ | ![img](../results/p50/result/unresolved/heatmap/rank5_id29_id37_0005.mp4.png)<br>`0.6515`<br>_id29_id37_00.._ |
| **Comp-q16-p0** | ![img](../results/q16-p0/result/unresolved/heatmap/rank1_id21_id19_0005.mp4.png)<br>`0.2383`<br>_id21_id19_00.._ | ![img](../results/q16-p0/result/unresolved/heatmap/rank2_id38_id34_0004.mp4.png)<br>`0.2238`<br>_id38_id34_00.._ | ![img](../results/q16-p0/result/unresolved/heatmap/rank3_id4_id6_0008.mp4.png)<br>`0.2188`<br>_id4_id6_0008.._ | ![img](../results/q16-p0/result/unresolved/heatmap/rank4_id34_id32_0007.mp4.png)<br>`0.1995`<br>_id34_id32_00.._ | ![img](../results/q16-p0/result/unresolved/heatmap/rank5_id13_id7_0000.mp4.png)<br>`0.1321`<br>_id13_id7_000.._ |
| **Comp-q16-p50** | ![img](../results/q16-p50/result/unresolved/heatmap/rank1_id31_id16_0002.mp4.png)<br>`0.7611`<br>_id31_id16_00.._ | ![img](../results/q16-p50/result/unresolved/heatmap/rank2_id34_id32_0007.mp4.png)<br>`0.7342`<br>_id34_id32_00.._ | ![img](../results/q16-p50/result/unresolved/heatmap/rank3_id30_id23_0007.mp4.png)<br>`0.6907`<br>_id30_id23_00.._ | ![img](../results/q16-p50/result/unresolved/heatmap/rank4_id30_id6_0007.mp4.png)<br>`0.6784`<br>_id30_id6_000.._ | ![img](../results/q16-p50/result/unresolved/heatmap/rank5_id29_id37_0005.mp4.png)<br>`0.6508`<br>_id29_id37_00.._ |
| **Comp-q8** | ![img](../results/q8/result/unresolved/heatmap/rank1_id7_id11_0007.mp4.png)<br>`0.8662`<br>_id7_id11_000.._ | ![img](../results/q8/result/unresolved/heatmap/rank2_id4_id6_0002.mp4.png)<br>`0.8490`<br>_id4_id6_0002.._ | ![img](../results/q8/result/unresolved/heatmap/rank3_id38_id33_0005.mp4.png)<br>`0.8249`<br>_id38_id33_00.._ | ![img](../results/q8/result/unresolved/heatmap/rank4_id0_id21_0000.mp4.png)<br>`0.8208`<br>_id0_id21_000.._ | ![img](../results/q8/result/unresolved/heatmap/rank5_id28_id4_0006.mp4.png)<br>`0.8090`<br>_id28_id4_000.._ |
| **Comp-q8-p50** | ![img](../results/q8-p50/result/unresolved/heatmap/rank1_id27_id25_0008.mp4.png)<br>`0.9288`<br>_id27_id25_00.._ | ![img](../results/q8-p50/result/unresolved/heatmap/rank2_id28_id4_0006.mp4.png)<br>`0.9201`<br>_id28_id4_000.._ | ![img](../results/q8-p50/result/unresolved/heatmap/rank3_id32_id33_0002.mp4.png)<br>`0.9162`<br>_id32_id33_00.._ | ![img](../results/q8-p50/result/unresolved/heatmap/rank4_id38_id33_0005.mp4.png)<br>`0.9144`<br>_id38_id33_00.._ | ![img](../results/q8-p50/result/unresolved/heatmap/rank5_id0_id1_0000.mp4.png)<br>`0.9093`<br>_id0_id1_0000.._ |



### 🚨 Worst Mismatches (Error Analysis)

These are the samples causing the biggest drop in ROC.

#### Top 5 False Positives (Real Videos labeled as Fake)
| Rank | baseline | p50 | q16-p0 | q16-p50 | q8 | q8-p50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 | **0.9808**<br>`00047.mp4` | **0.9966**<br>`00047.mp4` | **0.9809**<br>`00047.mp4` | **0.9966**<br>`00047.mp4` | **0.9999**<br>`00119.mp4` | **0.9993**<br>`00264.mp4` |
| #2 | **0.8435**<br>`id51_0001.mp4` | **0.9955**<br>`id51_0001.mp4` | **0.8418**<br>`id51_0001.mp4` | **0.9955**<br>`id51_0001.mp4` | **0.9999**<br>`00236.mp4` | **0.9989**<br>`00252.mp4` |
| #3 | **0.8011**<br>`00170.mp4` | **0.9937**<br>`00082.mp4` | **0.8003**<br>`00170.mp4` | **0.9936**<br>`00082.mp4` | **0.9999**<br>`id37_0004.mp4` | **0.9987**<br>`id13_0011.mp4` |
| #4 | **0.7185**<br>`id31_0003.mp4` | **0.9878**<br>`00256.mp4` | **0.7177**<br>`id31_0003.mp4` | **0.9876**<br>`00256.mp4` | **0.9998**<br>`00252.mp4` | **0.9984**<br>`00256.mp4` |
| #5 | **0.6559**<br>`id21_0009.mp4` | **0.9734**<br>`00264.mp4` | **0.6526**<br>`id21_0009.mp4` | **0.9728**<br>`00264.mp4` | **0.9998**<br>`00264.mp4` | **0.9984**<br>`00119.mp4` |

#### Top 5 False Negatives (Fake Videos labeled as Real)
| Rank | baseline | p50 | q16-p0 | q16-p50 | q8 | q8-p50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 | **0.0304**<br>`id29_id32_0000.mp4` | **0.3973**<br>`id9_id3_0009.mp4` | **0.0301**<br>`id29_id32_0000.mp4` | **0.3961**<br>`id9_id3_0009.mp4` | **0.3872**<br>`id10_id11_0001.mp4` | **0.7264**<br>`id29_id37_0005.mp4` |
| #2 | **0.1034**<br>`id30_id6_0007.mp4` | **0.5185**<br>`id29_id32_0000.mp4` | **0.1025**<br>`id30_id6_0007.mp4` | **0.5147**<br>`id29_id32_0000.mp4` | **0.6605**<br>`id29_id37_0005.mp4` | **0.8066**<br>`id10_id11_0001.mp4` |
| #3 | **0.1215**<br>`id33_id32_0006.mp4` | **0.5206**<br>`id33_id32_0006.mp4` | **0.1206**<br>`id33_id32_0006.mp4` | **0.5180**<br>`id33_id32_0006.mp4` | **0.7418**<br>`id49_id52_0007.mp4` | **0.8699**<br>`id0_id21_0000.mp4` |
| #4 | **0.1218**<br>`id30_id23_0007.mp4` | **0.5713**<br>`id38_id34_0004.mp4` | **0.1208**<br>`id30_id23_0007.mp4` | **0.5695**<br>`id38_id34_0004.mp4` | **0.7617**<br>`id10_id13_0001.mp4` | **0.9067**<br>`id17_id28_0001.mp4` |
| #5 | **0.1294**<br>`id9_id3_0009.mp4` | **0.6394**<br>`id16_id6_0001.mp4` | **0.1291**<br>`id9_id3_0009.mp4` | **0.6378**<br>`id16_id6_0001.mp4` | **0.7899**<br>`id0_id1_0000.mp4` | **0.9071**<br>`id13_id7_0012.mp4` |
