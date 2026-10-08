# Faithfulness results (deletion=blur, insertion=mean)

Raw AUC follows RISE (Petsiuk et al., 2018); AOPC-random is the AOPC (Samek et al., 2017) of the saliency ordering minus the random ordering, positive means more faithful than chance. Macro is the mean over the 7 classes.

| Model | Class | n | Deletion AUC (lower) | Deletion AOPC-random (+faithful) | Insertion AUC (higher) | Insertion AOPC-random (+faithful) |
|---|---|---:|---:|---:|---:|---:|
| **EfficientNet-B0** | akiec | 29 | 0.030 | +0.316 | 0.050 | +0.016 |
|  | bcc | 60 | 0.065 | +0.354 | 0.126 | +0.024 |
|  | bkl | 106 | 0.079 | +0.493 | 0.139 | -0.288 |
|  | df | 20 | 0.028 | +0.473 | 0.040 | -0.119 |
|  | mel | 60 | 0.080 | +0.447 | 0.101 | -0.059 |
|  | nv | 696 | 0.846 | -0.090 | 0.877 | +0.094 |
|  | vasc | 31 | 0.093 | +0.552 | 0.092 | -0.280 |
|  | **macro** | 1002 | 0.174 | +0.363 | 0.203 | -0.088 |
| **ViT-B/16** | akiec | 28 | 0.600 | -0.120 | 0.163 | +0.049 |
|  | bcc | 48 | 0.666 | -0.034 | 0.267 | +0.147 |
|  | bkl | 130 | 0.643 | -0.039 | 0.676 | +0.007 |
|  | df | 13 | 0.544 | +0.149 | 0.322 | +0.082 |
|  | mel | 77 | 0.250 | +0.301 | 0.352 | +0.115 |
|  | nv | 689 | 0.945 | +0.015 | 0.705 | +0.282 |
|  | vasc | 17 | 0.872 | -0.039 | 0.526 | +0.227 |
|  | **macro** | 1002 | 0.646 | +0.033 | 0.430 | +0.130 |
