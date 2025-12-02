# Deepfake Compression Report

## Configuration
* **Backbone:** b5

## Performance Table
| Model | AUC | Size (MB) | 
| :--- | :--- | :--- | 
| Baseline | 0.9549 | 108.79 |
| compressed_q16-p0 | 0.9548 | 54.60 |
| compressed_q16-p50 | 0.9334 | 54.60 |
| compressed_p50 | 0.9332 | 108.99 |
| compressed_q8 | 0.7588 | 27.80 |
| compressed_q8-p50 | 0.7119 | 27.80 |

## Analysis
Detailed heatmaps for resolved and unresolved samples. **Note:** Image paths are relative to the `results` folder.

### 🔹 Model: BASELINE
| Rank | ✅ Top Resolved (Detected Fake) | ❌ Top Unresolved (Missed Fake) |
| :---: | :--- | :--- |
| #1 | ![](../results/baseline/result/resolved/heatmap/rank1_id37_id3_0004.mp4.png)<br>**Score:** 0.9999<br>`id37_id3_0004.mp4` | ![](../results/baseline/result/unresolved/heatmap/rank1_id21_id19_0005.mp4.png)<br>**Score:** 0.2413<br>`id21_id19_0005.mp4` |
| #2 | ![](../results/baseline/result/resolved/heatmap/rank2_id46_id41_0000.mp4.png)<br>**Score:** 0.9999<br>`id46_id41_0000.mp4` | ![](../results/baseline/result/unresolved/heatmap/rank2_id38_id34_0004.mp4.png)<br>**Score:** 0.2252<br>`id38_id34_0004.mp4` |
| #3 | ![](../results/baseline/result/resolved/heatmap/rank3_id43_id40_0005.mp4.png)<br>**Score:** 0.9999<br>`id43_id40_0005.mp4` | ![](../results/baseline/result/unresolved/heatmap/rank3_id4_id6_0008.mp4.png)<br>**Score:** 0.2219<br>`id4_id6_0008.mp4` |

---
### 🔹 Model: COMPRESSED Q16-P0
| Rank | ✅ Top Resolved (Detected Fake) | ❌ Top Unresolved (Missed Fake) |
| :---: | :--- | :--- |
| #1 | ![](../results/q16-p0/result/resolved/heatmap/rank1_id37_id3_0004.mp4.png)<br>**Score:** 0.9999<br>`id37_id3_0004.mp4` | ![](../results/q16-p0/result/unresolved/heatmap/rank1_id21_id19_0005.mp4.png)<br>**Score:** 0.2383<br>`id21_id19_0005.mp4` |
| #2 | ![](../results/q16-p0/result/resolved/heatmap/rank2_id46_id41_0000.mp4.png)<br>**Score:** 0.9999<br>`id46_id41_0000.mp4` | ![](../results/q16-p0/result/unresolved/heatmap/rank2_id38_id34_0004.mp4.png)<br>**Score:** 0.2238<br>`id38_id34_0004.mp4` |
| #3 | ![](../results/q16-p0/result/resolved/heatmap/rank3_id43_id40_0005.mp4.png)<br>**Score:** 0.9999<br>`id43_id40_0005.mp4` | ![](../results/q16-p0/result/unresolved/heatmap/rank3_id4_id6_0008.mp4.png)<br>**Score:** 0.2188<br>`id4_id6_0008.mp4` |

---
### 🔹 Model: COMPRESSED Q16-P50
| Rank | ✅ Top Resolved (Detected Fake) | ❌ Top Unresolved (Missed Fake) |
| :---: | :--- | :--- |
| #1 | ![](../results/q16-p50/result/resolved/heatmap/rank1_id35_id31_0006.mp4.png)<br>**Score:** 0.9986<br>`id35_id31_0006.mp4` | ![](../results/q16-p50/result/unresolved/heatmap/rank1_id31_id16_0002.mp4.png)<br>**Score:** 0.7611<br>`id31_id16_0002.mp4` |
| #2 | ![](../results/q16-p50/result/resolved/heatmap/rank2_id37_id28_0007.mp4.png)<br>**Score:** 0.9982<br>`id37_id28_0007.mp4` | ![](../results/q16-p50/result/unresolved/heatmap/rank2_id34_id32_0007.mp4.png)<br>**Score:** 0.7342<br>`id34_id32_0007.mp4` |
| #3 | ![](../results/q16-p50/result/resolved/heatmap/rank3_id21_id20_0006.mp4.png)<br>**Score:** 0.9981<br>`id21_id20_0006.mp4` | ![](../results/q16-p50/result/unresolved/heatmap/rank3_id30_id23_0007.mp4.png)<br>**Score:** 0.6907<br>`id30_id23_0007.mp4` |

---
### 🔹 Model: COMPRESSED P50
| Rank | ✅ Top Resolved (Detected Fake) | ❌ Top Unresolved (Missed Fake) |
| :---: | :--- | :--- |
| #1 | ![](../results/p50/result/resolved/heatmap/rank1_id35_id31_0006.mp4.png)<br>**Score:** 0.9986<br>`id35_id31_0006.mp4` | ![](../results/p50/result/unresolved/heatmap/rank1_id31_id16_0002.mp4.png)<br>**Score:** 0.7619<br>`id31_id16_0002.mp4` |
| #2 | ![](../results/p50/result/resolved/heatmap/rank2_id37_id28_0007.mp4.png)<br>**Score:** 0.9982<br>`id37_id28_0007.mp4` | ![](../results/p50/result/unresolved/heatmap/rank2_id34_id32_0007.mp4.png)<br>**Score:** 0.7375<br>`id34_id32_0007.mp4` |
| #3 | ![](../results/p50/result/resolved/heatmap/rank3_id21_id20_0006.mp4.png)<br>**Score:** 0.9981<br>`id21_id20_0006.mp4` | ![](../results/p50/result/unresolved/heatmap/rank3_id30_id23_0007.mp4.png)<br>**Score:** 0.6920<br>`id30_id23_0007.mp4` |

---
### 🔹 Model: COMPRESSED Q8
| Rank | ✅ Top Resolved (Detected Fake) | ❌ Top Unresolved (Missed Fake) |
| :---: | :--- | :--- |
| #1 | ![](../results/q8/result/resolved/heatmap/rank1_id46_id41_0000.mp4.png)<br>**Score:** 1.0000<br>`id46_id41_0000.mp4` | ![](../results/q8/result/unresolved/heatmap/rank1_id7_id11_0007.mp4.png)<br>**Score:** 0.8662<br>`id7_id11_0007.mp4` |
| #2 | ![](../results/q8/result/resolved/heatmap/rank2_id2_id0_0008.mp4.png)<br>**Score:** 1.0000<br>`id2_id0_0008.mp4` | ![](../results/q8/result/unresolved/heatmap/rank2_id4_id6_0002.mp4.png)<br>**Score:** 0.8490<br>`id4_id6_0002.mp4` |
| #3 | ![](../results/q8/result/resolved/heatmap/rank3_id1_id2_0002.mp4.png)<br>**Score:** 1.0000<br>`id1_id2_0002.mp4` | ![](../results/q8/result/unresolved/heatmap/rank3_id38_id33_0005.mp4.png)<br>**Score:** 0.8249<br>`id38_id33_0005.mp4` |

---
### 🔹 Model: COMPRESSED Q8-P50
| Rank | ✅ Top Resolved (Detected Fake) | ❌ Top Unresolved (Missed Fake) |
| :---: | :--- | :--- |
| #1 | ![](../results/q8-p50/result/resolved/heatmap/rank1_id1_id2_0007.mp4.png)<br>**Score:** 0.9998<br>`id1_id2_0007.mp4` | ![](../results/q8-p50/result/unresolved/heatmap/rank1_id27_id25_0008.mp4.png)<br>**Score:** 0.9288<br>`id27_id25_0008.mp4` |
| #2 | ![](../results/q8-p50/result/resolved/heatmap/rank2_id17_id2_0000.mp4.png)<br>**Score:** 0.9995<br>`id17_id2_0000.mp4` | ![](../results/q8-p50/result/unresolved/heatmap/rank2_id28_id4_0006.mp4.png)<br>**Score:** 0.9201<br>`id28_id4_0006.mp4` |
| #3 | ![](../results/q8-p50/result/resolved/heatmap/rank3_id2_id26_0001.mp4.png)<br>**Score:** 0.9994<br>`id2_id26_0001.mp4` | ![](../results/q8-p50/result/unresolved/heatmap/rank3_id32_id33_0002.mp4.png)<br>**Score:** 0.9162<br>`id32_id33_0002.mp4` |

---
