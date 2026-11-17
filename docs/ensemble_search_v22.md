# Trust List 2.2 ensemble search (4–10 checkpoints)

Trust List: `trust-list-2.2-v0005` from `docs/trust-list-2.2-v0005.csv`; SHA-256 `297c80091322b62c8e706edba794a20554306368a2e59b309805e3bb418398fe`.
Reproduce with `python -m src.ensemble_search_v22`.
The candidate pool includes all **22** eligible checkpoints. Probabilities were loaded from cached full-V0 prediction files only; no checkpoints were loaded and no inference or training ran.
Selection used the existing five prompt-group folds. Each fold's selected members and optimized weights were fitted on its other four folds, then applied once to held-out rows. The Trust List roster/checkpoint history and prior checkpoint selection may have seen V0, so cross-fitting only evaluates this ensemble-selection layer; it is not an independent Kaggle estimate.
Optimized weights use the existing ensemble_factory regularized objective: training-fold log loss plus L2 shrinkage toward equal weights, nonnegative simplex constraints. Search uses deterministic beam expansion and one-member local swaps.

## Reference and search settings

EQUAL9 full-V0 balanced accuracy: **65.3898%** (target reference 65.3898%). Cross-fitted fixed EQUAL9 balanced accuracy: **65.3898%**.
Beam width 24; local-search seeds 2; swap passes 1; sizes 4–10; top 3 per method and size.

## Cross-fitted results by size and rank

Rank refers to a configuration selected using only that fold's fitting rows. Thus member IDs and optimized weights can vary across folds; exact selections are in `fold_selections.json`. Metrics pool the five held-out prediction partitions.

| Size | Weight method | Fit-fold rank | OOF balanced accuracy | OOF accuracy | OOF log loss | Fold BA SD | Mean member Jaccard | Δ BA vs EQUAL9 | Corrected / introduced vs EQUAL9 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | equal | 1 | 61.6073% | 61.6154% | 1.113345 | 2.529 pp | 0.840 | -3.783 pp | 195 / 297 |
| 4 | equal | 2 | 60.7159% | 60.7262% | 1.119397 | 1.600 pp | 0.560 | -4.674 pp | 177 / 303 |
| 4 | equal | 3 | 60.3828% | 60.3927% | 1.126035 | 2.596 pp | 0.507 | -5.007 pp | 150 / 285 |
| 5 | equal | 1 | 60.6060% | 60.6150% | 1.150337 | 2.570 pp | 0.474 | -4.784 pp | 165 / 294 |
| 5 | equal | 2 | 60.6423% | 60.6521% | 1.128984 | 1.711 pp | 0.353 | -4.747 pp | 162 / 290 |
| 5 | equal | 3 | 60.1613% | 60.1704% | 1.106297 | 2.273 pp | 0.488 | -5.228 pp | 154 / 295 |
| 6 | equal | 1 | 60.9028% | 60.9114% | 1.130018 | 2.407 pp | 0.478 | -4.487 pp | 168 / 289 |
| 6 | equal | 2 | 61.2744% | 61.2820% | 1.094879 | 2.002 pp | 0.463 | -4.115 pp | 150 / 261 |
| 6 | equal | 3 | 61.1646% | 61.1708% | 1.125727 | 1.548 pp | 0.425 | -4.225 pp | 164 / 278 |
| 7 | equal | 1 | 61.0139% | 61.0226% | 1.073553 | 2.613 pp | 0.582 | -4.376 pp | 160 / 278 |
| 7 | equal | 2 | 60.6083% | 60.6150% | 1.080406 | 1.875 pp | 0.517 | -4.782 pp | 157 / 286 |
| 7 | equal | 3 | 61.3865% | 61.3931% | 1.078898 | 2.234 pp | 0.504 | -4.003 pp | 165 / 273 |
| 8 | equal | 1 | 61.0524% | 61.0597% | 1.092632 | 2.390 pp | 0.533 | -4.337 pp | 162 / 279 |
| 8 | equal | 2 | 61.5712% | 61.5784% | 1.096588 | 1.813 pp | 0.453 | -3.819 pp | 152 / 255 |
| 8 | equal | 3 | 61.4233% | 61.4302% | 1.080303 | 2.566 pp | 0.715 | -3.966 pp | 161 / 268 |
| 9 | equal | 1 | 61.1643% | 61.1708% | 1.080128 | 2.669 pp | 0.504 | -4.225 pp | 157 / 271 |
| 9 | equal | 2 | 61.1651% | 61.1708% | 1.086135 | 2.284 pp | 0.625 | -4.225 pp | 157 / 271 |
| 9 | equal | 3 | 61.6473% | 61.6525% | 1.073651 | 2.420 pp | 0.659 | -3.742 pp | 165 / 266 |
| 10 | equal | 1 | 61.3497% | 61.3561% | 1.083404 | 2.472 pp | 0.646 | -4.040 pp | 159 / 268 |
| 10 | equal | 2 | 61.5727% | 61.5784% | 1.081448 | 2.245 pp | 0.745 | -3.817 pp | 157 / 260 |
| 10 | equal | 3 | 61.4246% | 61.4302% | 1.087700 | 2.723 pp | 0.707 | -3.965 pp | 151 / 258 |
| 4 | optimized | 1 | 62.0172% | 62.0230% | 1.058947 | 2.842 pp | 0.840 | -3.373 pp | 181 / 272 |
| 4 | optimized | 2 | 61.6075% | 61.6154% | 1.059881 | 1.849 pp | 0.573 | -3.782 pp | 184 / 286 |
| 4 | optimized | 3 | 60.8998% | 60.9114% | 1.077551 | 2.645 pp | 0.341 | -4.490 pp | 178 / 299 |
| 5 | optimized | 1 | 61.5705% | 61.5784% | 1.069811 | 2.409 pp | 0.448 | -3.819 pp | 174 / 277 |
| 5 | optimized | 2 | 60.9038% | 60.9114% | 1.061425 | 2.662 pp | 0.432 | -4.486 pp | 177 / 298 |
| 5 | optimized | 3 | 61.2739% | 61.2820% | 1.053792 | 2.027 pp | 0.476 | -4.116 pp | 177 / 288 |
| 6 | optimized | 1 | 61.8685% | 61.8748% | 1.062397 | 2.703 pp | 0.468 | -3.521 pp | 171 / 266 |
| 6 | optimized | 2 | 61.8307% | 61.8377% | 1.056704 | 2.147 pp | 0.483 | -3.559 pp | 185 / 281 |
| 6 | optimized | 3 | 62.1265% | 62.1341% | 1.062581 | 2.762 pp | 0.401 | -3.263 pp | 176 / 264 |
| 7 | optimized | 1 | 61.4959% | 61.5043% | 1.054444 | 2.309 pp | 0.429 | -3.894 pp | 177 / 282 |
| 7 | optimized | 2 | 61.2749% | 61.2820% | 1.062899 | 2.302 pp | 0.362 | -4.115 pp | 164 / 275 |
| 7 | optimized | 3 | 61.8302% | 61.8377% | 1.049918 | 2.111 pp | 0.488 | -3.560 pp | 173 / 269 |
| 8 | optimized | 1 | 61.3112% | 61.3190% | 1.050304 | 2.784 pp | 0.447 | -4.079 pp | 165 / 275 |
| 8 | optimized | 2 | 61.7564% | 61.7636% | 1.046375 | 2.111 pp | 0.438 | -3.633 pp | 171 / 269 |
| 8 | optimized | 3 | 61.1618% | 61.1708% | 1.053211 | 2.291 pp | 0.315 | -4.228 pp | 169 / 283 |
| 9 | optimized | 1 | 61.6450% | 61.6525% | 1.045121 | 2.922 pp | 0.461 | -3.745 pp | 168 / 269 |
| 9 | optimized | 2 | 61.2016% | 61.2079% | 1.047041 | 2.878 pp | 0.521 | -4.188 pp | 160 / 273 |
| 9 | optimized | 3 | 61.1263% | 61.1338% | 1.047629 | 2.632 pp | 0.381 | -4.263 pp | 165 / 280 |
| 10 | optimized | 1 | 61.7576% | 61.7636% | 1.043560 | 2.870 pp | 0.495 | -3.632 pp | 166 / 264 |
| 10 | optimized | 2 | 61.6468% | 61.6525% | 1.042854 | 3.042 pp | 0.570 | -3.743 pp | 166 / 267 |
| 10 | optimized | 3 | 61.1275% | 61.1338% | 1.046309 | 3.072 pp | 0.479 | -4.262 pp | 158 / 273 |

## Full-V0 selected configurations (optimistic diagnostics)

Each configuration below was selected and scored on all V0 rows. Use these exact IDs and weights only as development candidates; scores are optimistic after selection.

### 4 checkpoints

| Method | Rank | Checkpoint IDs | Weights | Full-V0 BA | Accuracy | Log loss | Corrected / introduced | Δ BA vs EQUAL9 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| equal | 1 | N0072, N0074, N0081, N0105 | N0072:0.2500, N0074:0.2500, N0081:0.2500, N0105:0.2500 | 61.8688% | 61.8748% | 1.107604 | 190 / 285 | -3.521 pp |
| equal | 2 | N0070, N0072, N0074, N0081 | N0070:0.2500, N0072:0.2500, N0074:0.2500, N0081:0.2500 | 61.6383% | 61.6525% | 1.115573 | 200 / 301 | -3.751 pp |
| equal | 3 | N0071, N0072, N0081, N0106 | N0071:0.2500, N0072:0.2500, N0081:0.2500, N0106:0.2500 | 61.3855% | 61.3931% | 1.136367 | 167 / 275 | -4.004 pp |
| optimized | 1 | N0074, N0076, N0081, N0105 | N0074:0.2677, N0076:0.1374, N0081:0.3846, N0105:0.2104 | 62.2769% | 62.2823% | 1.058965 | 192 / 276 | -3.113 pp |
| optimized | 2 | N0071, N0074, N0081, N0105 | N0071:0.1270, N0074:0.2733, N0081:0.3892, N0105:0.2106 | 62.1648% | 62.1712% | 1.062463 | 180 / 267 | -3.225 pp |
| optimized | 3 | N0074, N0079, N0081, N0105 | N0074:0.2756, N0079:0.0853, N0081:0.4042, N0105:0.2348 | 62.0900% | 62.0971% | 1.065567 | 196 / 285 | -3.300 pp |

### 5 checkpoints

| Method | Rank | Checkpoint IDs | Weights | Full-V0 BA | Accuracy | Log loss | Corrected / introduced | Δ BA vs EQUAL9 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| equal | 1 | N0071, N0072, N0073, N0080, N0081 | N0071:0.2000, N0072:0.2000, N0073:0.2000, N0080:0.2000, N0081:0.2000 | 61.7934% | 61.8007% | 1.180661 | 156 / 253 | -3.596 pp |
| equal | 2 | N0072, N0073, N0074, N0081, N0105 | N0072:0.2000, N0073:0.2000, N0074:0.2000, N0081:0.2000, N0105:0.2000 | 61.6833% | 61.6895% | 1.179335 | 180 / 280 | -3.706 pp |
| equal | 3 | N0072, N0074, N0080, N0081, N0105 | N0072:0.2000, N0074:0.2000, N0080:0.2000, N0081:0.2000, N0105:0.2000 | 61.5720% | 61.5784% | 1.093615 | 188 / 291 | -3.818 pp |
| optimized | 1 | N0071, N0074, N0078, N0081, N0105 | N0071:0.1180, N0074:0.2492, N0078:0.0598, N0081:0.3691, N0105:0.2039 | 62.4609% | 62.4676% | 1.060265 | 187 / 266 | -2.929 pp |
| optimized | 2 | N0074, N0078, N0079, N0081, N0105 | N0074:0.2457, N0078:0.0681, N0079:0.0812, N0081:0.3800, N0105:0.2249 | 62.4601% | 62.4676% | 1.062809 | 193 / 272 | -2.930 pp |
| optimized | 3 | N0071, N0074, N0079, N0081, N0105 | N0071:0.1093, N0074:0.2556, N0079:0.0610, N0081:0.3765, N0105:0.1977 | 62.2762% | 62.2823% | 1.060720 | 181 / 265 | -3.114 pp |

### 6 checkpoints

| Method | Rank | Checkpoint IDs | Weights | Full-V0 BA | Accuracy | Log loss | Corrected / introduced | Δ BA vs EQUAL9 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| equal | 1 | N0071, N0072, N0073, N0081, N0105, N0106 | N0071:0.1667, N0072:0.1667, N0073:0.1667, N0081:0.1667, N0105:0.1667, N0106:0.1667 | 62.0905% | 62.0971% | 1.146734 | 165 / 254 | -3.299 pp |
| equal | 2 | N0062, N0071, N0072, N0080, N0081, N0106 | N0062:0.1667, N0071:0.1667, N0072:0.1667, N0080:0.1667, N0081:0.1667, N0106:0.1667 | 62.0505% | 62.0600% | 1.098492 | 166 / 256 | -3.339 pp |
| equal | 3 | N0065, N0071, N0072, N0080, N0081, N0106 | N0065:0.1667, N0071:0.1667, N0072:0.1667, N0080:0.1667, N0081:0.1667, N0106:0.1667 | 62.0142% | 62.0230% | 1.085844 | 170 / 261 | -3.376 pp |
| optimized | 1 | N0071, N0074, N0078, N0081, N0105, N0106 | N0071:0.1116, N0074:0.2160, N0078:0.0505, N0081:0.3583, N0105:0.2019, N0106:0.0617 | 62.6831% | 62.6899% | 1.058828 | 179 / 252 | -2.707 pp |
| optimized | 2 | N0072, N0074, N0078, N0079, N0081, N0105 | N0072:0.0354, N0074:0.2222, N0078:0.0717, N0079:0.0834, N0081:0.3689, N0105:0.2185 | 62.6453% | 62.6528% | 1.055602 | 194 / 268 | -2.744 pp |
| optimized | 3 | N0071, N0074, N0078, N0080, N0081, N0105 | N0071:0.1140, N0074:0.2317, N0078:0.0168, N0080:0.0882, N0081:0.3520, N0105:0.1973 | 62.6092% | 62.6158% | 1.058565 | 184 / 259 | -2.781 pp |

### 7 checkpoints

| Method | Rank | Checkpoint IDs | Weights | Full-V0 BA | Accuracy | Log loss | Corrected / introduced | Δ BA vs EQUAL9 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| equal | 1 | N0063, N0071, N0072, N0080, N0081, N0105, N0106 | N0063:0.1429, N0071:0.1429, N0072:0.1429, N0080:0.1429, N0081:0.1429, N0105:0.1429, N0106:0.1429 | 62.3137% | 62.3194% | 1.074915 | 161 / 244 | -3.076 pp |
| equal | 2 | N0065, N0071, N0072, N0074, N0081, N0105, N0106 | N0065:0.1429, N0071:0.1429, N0072:0.1429, N0074:0.1429, N0081:0.1429, N0105:0.1429, N0106:0.1429 | 62.2018% | 62.2082% | 1.070830 | 165 / 251 | -3.188 pp |
| equal | 3 | N0065, N0071, N0072, N0080, N0081, N0105, N0106 | N0065:0.1429, N0071:0.1429, N0072:0.1429, N0080:0.1429, N0081:0.1429, N0105:0.1429, N0106:0.1429 | 62.1641% | 62.1712% | 1.069302 | 162 / 249 | -3.226 pp |
| optimized | 1 | N0071, N0072, N0074, N0078, N0081, N0105, N0106 | N0071:0.1142, N0072:0.0358, N0074:0.1918, N0078:0.0538, N0081:0.3466, N0105:0.1954, N0106:0.0624 | 62.7939% | 62.8010% | 1.051623 | 178 / 248 | -2.596 pp |
| optimized | 2 | N0071, N0074, N0077, N0078, N0081, N0105, N0106 | N0071:0.1067, N0074:0.2024, N0077:0.0247, N0078:0.0517, N0081:0.3532, N0105:0.1993, N0106:0.0621 | 62.6833% | 62.6899% | 1.056040 | 181 / 254 | -2.706 pp |
| optimized | 3 | N0071, N0072, N0074, N0080, N0081, N0105, N0106 | N0071:0.1115, N0072:0.0355, N0074:0.1777, N0080:0.0953, N0081:0.3315, N0105:0.1898, N0106:0.0587 | 62.6831% | 62.6899% | 1.049899 | 179 / 252 | -2.707 pp |

### 8 checkpoints

| Method | Rank | Checkpoint IDs | Weights | Full-V0 BA | Accuracy | Log loss | Corrected / introduced | Δ BA vs EQUAL9 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| equal | 1 | N0065, N0071, N0072, N0074, N0080, N0081, N0105, N0106 | N0065:0.1250, N0071:0.1250, N0072:0.1250, N0074:0.1250, N0080:0.1250, N0081:0.1250, N0105:0.1250, N0106:0.1250 | 62.3865% | 62.3935% | 1.064498 | 158 / 239 | -3.003 pp |
| equal | 2 | N0065, N0071, N0072, N0073, N0080, N0081, N0105, N0106 | N0065:0.1250, N0071:0.1250, N0072:0.1250, N0073:0.1250, N0080:0.1250, N0081:0.1250, N0105:0.1250, N0106:0.1250 | 62.3495% | 62.3564% | 1.110340 | 161 / 243 | -3.040 pp |
| equal | 3 | N0068, N0071, N0072, N0074, N0080, N0081, N0105, N0106 | N0068:0.1250, N0071:0.1250, N0072:0.1250, N0074:0.1250, N0080:0.1250, N0081:0.1250, N0105:0.1250, N0106:0.1250 | 62.2406% | 62.2453% | 1.071158 | 159 / 244 | -3.149 pp |
| optimized | 1 | N0071, N0072, N0073, N0074, N0078, N0081, N0105, N0106 | N0071:0.1142, N0072:0.0358, N0073:0.0000, N0074:0.1918, N0078:0.0538, N0081:0.3466, N0105:0.1954, N0106:0.0624 | 62.8310% | 62.8381% | 1.051623 | 179 / 248 | -2.559 pp |
| optimized | 2 | N0071, N0072, N0074, N0077, N0078, N0081, N0105, N0106 | N0071:0.1135, N0072:0.0346, N0074:0.1908, N0077:0.0036, N0078:0.0539, N0081:0.3463, N0105:0.1951, N0106:0.0623 | 62.7939% | 62.8010% | 1.051644 | 178 / 248 | -2.596 pp |
| optimized | 3 | N0071, N0072, N0074, N0076, N0078, N0079, N0081, N0105 | N0071:0.0757, N0072:0.0330, N0074:0.1972, N0076:0.0926, N0078:0.0552, N0079:0.0516, N0081:0.3313, N0105:0.1634 | 62.7216% | 62.7269% | 1.047405 | 171 / 243 | -2.668 pp |

### 9 checkpoints

| Method | Rank | Checkpoint IDs | Weights | Full-V0 BA | Accuracy | Log loss | Corrected / introduced | Δ BA vs EQUAL9 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| equal | 1 | N0065, N0071, N0072, N0073, N0074, N0080, N0081, N0105, N0106 | N0065:0.1111, N0071:0.1111, N0072:0.1111, N0073:0.1111, N0074:0.1111, N0080:0.1111, N0081:0.1111, N0105:0.1111, N0106:0.1111 | 62.4601% | 62.4676% | 1.101936 | 154 / 233 | -2.930 pp |
| equal | 2 | N0065, N0068, N0070, N0071, N0072, N0074, N0080, N0081, N0105 | N0065:0.1111, N0068:0.1111, N0070:0.1111, N0071:0.1111, N0072:0.1111, N0074:0.1111, N0080:0.1111, N0081:0.1111, N0105:0.1111 | 62.3888% | 62.3935% | 1.056207 | 163 / 244 | -3.001 pp |
| equal | 3 | N0068, N0071, N0072, N0073, N0074, N0080, N0081, N0105, N0106 | N0068:0.1111, N0071:0.1111, N0072:0.1111, N0073:0.1111, N0074:0.1111, N0080:0.1111, N0081:0.1111, N0105:0.1111, N0106:0.1111 | 62.3515% | 62.3564% | 1.103433 | 154 / 236 | -3.038 pp |
| optimized | 1 | N0065, N0071, N0072, N0074, N0076, N0078, N0081, N0105, N0106 | N0065:0.1307, N0071:0.0829, N0072:0.0194, N0074:0.1302, N0076:0.0795, N0078:0.0482, N0081:0.3066, N0105:0.1437, N0106:0.0588 | 62.8680% | 62.8751% | 1.036171 | 175 / 243 | -2.522 pp |
| optimized | 2 | N0071, N0072, N0074, N0076, N0078, N0079, N0081, N0105, N0106 | N0071:0.0715, N0072:0.0319, N0074:0.1674, N0076:0.0877, N0078:0.0479, N0079:0.0529, N0081:0.3216, N0105:0.1629, N0106:0.0564 | 62.8322% | 62.8381% | 1.046323 | 178 / 247 | -2.558 pp |
| optimized | 3 | N0067, N0071, N0072, N0074, N0078, N0079, N0081, N0105, N0106 | N0067:0.0608, N0071:0.0922, N0072:0.0335, N0074:0.1612, N0078:0.0464, N0079:0.0465, N0081:0.3340, N0105:0.1668, N0106:0.0585 | 62.7959% | 62.8010% | 1.044972 | 174 / 244 | -2.594 pp |

### 10 checkpoints

| Method | Rank | Checkpoint IDs | Weights | Full-V0 BA | Accuracy | Log loss | Corrected / introduced | Δ BA vs EQUAL9 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| equal | 1 | N0070, N0071, N0072, N0074, N0077, N0079, N0080, N0081, N0105, N0106 | N0070:0.1000, N0071:0.1000, N0072:0.1000, N0074:0.1000, N0077:0.1000, N0079:0.1000, N0080:0.1000, N0081:0.1000, N0105:0.1000, N0106:0.1000 | 62.4626% | 62.4676% | 1.081260 | 166 / 245 | -2.927 pp |
| equal | 2 | N0068, N0070, N0071, N0072, N0074, N0077, N0080, N0081, N0105, N0106 | N0068:0.1000, N0070:0.1000, N0071:0.1000, N0072:0.1000, N0074:0.1000, N0077:0.1000, N0080:0.1000, N0081:0.1000, N0105:0.1000, N0106:0.1000 | 62.4261% | 62.4305% | 1.077770 | 162 / 242 | -2.964 pp |
| equal | 3 | N0068, N0070, N0071, N0072, N0074, N0079, N0080, N0081, N0105, N0106 | N0068:0.1000, N0070:0.1000, N0071:0.1000, N0072:0.1000, N0074:0.1000, N0079:0.1000, N0080:0.1000, N0081:0.1000, N0105:0.1000, N0106:0.1000 | 62.4253% | 62.4305% | 1.063197 | 166 / 246 | -2.964 pp |
| optimized | 1 | N0067, N0071, N0072, N0074, N0078, N0079, N0080, N0081, N0105, N0106 | N0067:0.0574, N0071:0.0907, N0072:0.0333, N0074:0.1516, N0078:0.0156, N0079:0.0443, N0080:0.0652, N0081:0.3227, N0105:0.1641, N0106:0.0554 | 62.9068% | 62.9122% | 1.044044 | 171 / 238 | -2.483 pp |
| optimized | 2 | N0071, N0072, N0074, N0076, N0077, N0078, N0079, N0081, N0105, N0106 | N0071:0.0712, N0072:0.0313, N0074:0.1670, N0076:0.0877, N0077:0.0015, N0078:0.0479, N0079:0.0528, N0081:0.3214, N0105:0.1627, N0106:0.0564 | 62.8693% | 62.8751% | 1.046334 | 178 / 246 | -2.521 pp |
| optimized | 3 | N0065, N0071, N0072, N0074, N0076, N0077, N0078, N0081, N0105, N0106 | N0065:0.1307, N0071:0.0829, N0072:0.0194, N0074:0.1302, N0076:0.0795, N0077:0.0000, N0078:0.0482, N0081:0.3066, N0105:0.1437, N0106:0.0588 | 62.8680% | 62.8751% | 1.036171 | 175 / 243 | -2.522 pp |

## Best three distinct cross-fitted candidates

Candidate selection is ranked by pooled OOF balanced accuracy across all three fit-fold ranks. Among candidates within 0.5 percentage points of the leader, prediction disagreement is used to prefer complementary procedures. Diversity does not alter the reported accuracy ranking.

| Rank | Size | Method | Fit-fold rank | OOF BA | OOF accuracy | OOF log loss | Disagreement vs prior selected |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 6 | optimized | 3 | 62.1265% | 62.1341% | 1.062581 | — |
|  |  | Per-fold selected members: [["N0065","N0074","N0076","N0078","N0081","N0105"],["N0070","N0071","N0072","N0080","N0081","N0106"],["N0071","N0072","N0074","N0078","N0081","N0105"],["N0071","N0074","N0075","N0080","N0081","N0105"],["N0071","N0072","N0080","N0081","N0105","N0106"]] |  |  |  |  |  |
| 2 | 4 | optimized | 1 | 62.0172% | 62.0230% | 1.058947 | 10.52% |
|  |  | Per-fold selected members: [["N0074","N0076","N0081","N0105"],["N0071","N0074","N0081","N0105"],["N0074","N0076","N0081","N0105"],["N0074","N0076","N0081","N0105"],["N0074","N0076","N0081","N0105"]] |  |  |  |  |  |
| 3 | 6 | optimized | 1 | 61.8685% | 61.8748% | 1.062397 | 8.23% |
|  |  | Per-fold selected members: [["N0067","N0071","N0080","N0081","N0105","N0106"],["N0071","N0072","N0080","N0081","N0105","N0106"],["N0071","N0073","N0074","N0078","N0081","N0105"],["N0071","N0072","N0074","N0078","N0081","N0105"],["N0072","N0074","N0078","N0079","N0081","N0105"]] |  |  |  |  |  |

## Per-label metrics

Per-label precision, recall and F1 are stored in `per_label_metrics.csv` for every pooled OOF rank/size/method row and every full-V0 top-three configuration. Prediction and confusion matrices are stored in the JSON metrics and Parquet outputs.

## Assessment

- **Best cross-fitted procedure:** optimized 6-checkpoint fit-fold rank 3 at 62.1265% OOF BA.
- **Best equal-weight procedure:** equal 9-checkpoint fit-fold rank 3 at 61.6473% OOF BA.
- **Best optimized-weight procedure:** optimized 6-checkpoint fit-fold rank 3 at 62.1265% OOF BA.
- **Best full-V0 diagnostic:** optimized 10-checkpoint rank 1 at 62.9068% BA (-2.483 pp vs EQUAL9; optimistic diagnostic).
- **Compared with EQUAL9:** Best cross-fitted candidate changes balanced accuracy by -3.263 pp versus fixed EQUAL9 OOF (65.3898%); full-V0 baseline is 65.3898%.
- **Kaggle recommendation:** No candidate merits submission from this search: the best is optimized 6 at 62.1265% OOF BA, -3.263 pp below EQUAL9.
No Kaggle submission was made. A submission would be a follow-up decision; full-V0 diagnostics are optimistic, and the cross-fit procedures select different members in different folds.

## Reproducibility outputs

- `experiment_config.json`: Trust List version/hash, candidate IDs and probability-cache provenance.
- `fold_selections.json`: top-three fold-specific members and weights for every size and method.
- `full_v0_selections.json`: exact full-V0 top-three configurations and weights.
- `metrics.json` and `metrics.csv`: cross-fitted and full-V0 scores, corrections, errors and per-label metrics.
- `folds/fold_XX.npz`: held-out probabilities from fold-specific selections.
- `predictions_oof.parquet`, `predictions_full_v0.parquet`: row-aligned reproducible predictions and probabilities.
- `fold_assignments.csv`: reused five prompt-group fold IDs.
