# ORACLE04 — Top-1 vs Top-2 Decision Model

Validation: 2,699 V0 responses, 150 prompt groups, canonical 18-class order. Base probabilities and checkpoint predictions are cached; only CPU meta-classifiers were fitted. This is a development-set diagnostic, not an independent Kaggle estimate.

## Baseline and top-two opportunity

| System | Accuracy | Balanced accuracy | Log loss |
|---|---:|---:|---:|
| ORACLE02 | 67.173% | 67.167% | 1.19917 |
| ORACLE03_top2 | 67.506% | 67.500% | 1.19450 |
| H09 | 65.506% | 65.500% | 1.10515 |
| EQUAL9 | 65.395% | 65.390% | 1.13208 |
| top2_oracle | 84.216% | 84.213% | 5.68899 |

Among the saved ORACLE02 rows: top-1 is correct on 1,813; top-1 is wrong and top-2 correct on 460; neither is correct on 426. Top-2 union coverage is 84.22%; top-3 coverage is 91.14%. The oracle result is a ground-truth upper bound.

| Group | Eligible rows | Top-1 correct | Top-2 correct, top-1 wrong | Neither correct |
|---|---:|---:|---:|---:|
| A: gemini-3.7-flash, gemini-3.8-flash | 259 | 146 | 89 | 24 |
| B: claude-fable-5, glm-5.3 | 144 | 59 | 39 | 46 |

## Cross-fitted results

Thresholds maximize balanced-accuracy improvement on training-only inner-fold predictions. A no-swap decision is selected unless a threshold improves training-fold balanced accuracy. Swap precision is corrected / (corrected + introduced). Bootstrap intervals resample prompt groups (1,000 replicates) and measure the BA delta from ORACLE02.

| Strategy | Accuracy | Balanced accuracy | Log loss | Triggered | Trigger % | Corrected / introduced | Correction precision | Target precision / recall | Net correct | 95% cluster-bootstrap ΔBA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| combined_logistic_core | 71.175% | 71.169% | 1.18568 | 404 | 14.97% | 223 / 115 | 65.98% | 55.20% / 48.48% | +108 | [+2.706, +5.371] pp |
| global_logistic_core | 71.175% | 71.168% | 1.18470 | 431 | 15.97% | 229 / 121 | 65.43% | 53.13% / 49.78% | +108 | [+2.630, +5.338] pp |
| global_catboost_core | 70.804% | 70.799% | 1.18875 | 336 | 12.45% | 196 / 98 | 66.67% | 58.33% / 42.61% | +98 | [+2.333, +4.852] pp |
| global_catboost_core_stylometry | 70.693% | 70.687% | 1.18727 | 316 | 11.71% | 184 / 89 | 67.40% | 58.23% / 40.00% | +95 | [+2.260, +4.780] pp |
| global_catboost_core_embedding | 70.693% | 70.687% | 1.18886 | 264 | 9.78% | 165 / 70 | 70.21% | 62.50% / 35.87% | +95 | [+2.221, +4.669] pp |
| global_logistic_core_stylometry | 70.396% | 70.391% | 1.18359 | 372 | 13.78% | 191 / 104 | 64.75% | 51.34% / 41.52% | +87 | [+1.776, +4.482] pp |
| combined_catboost_core | 70.322% | 70.317% | 1.19150 | 390 | 14.45% | 210 / 125 | 62.69% | 53.85% / 45.65% | +85 | [+1.853, +4.486] pp |
| global_catboost_core_stylometry_embedding | 70.285% | 70.278% | 1.19125 | 313 | 11.60% | 180 / 96 | 65.22% | 57.51% / 39.13% | +84 | [+1.852, +4.259] pp |
| global_logistic_core_embedding | 70.063% | 70.058% | 1.18826 | 266 | 9.86% | 147 / 69 | 68.06% | 55.26% / 31.96% | +78 | [+1.816, +3.926] pp |
| global_logistic_core_stylometry_embedding | 69.618% | 69.612% | 1.18995 | 274 | 10.15% | 140 / 74 | 65.42% | 51.09% / 30.43% | +66 | [+1.333, +3.486] pp |
| group_A_logistic_core | 68.247% | 68.241% | 1.19823 | 79 | 2.93% | 52 / 23 | 69.33% | 65.82% / 11.30% | +29 | [+0.519, +1.630] pp |
| group_A_catboost_core | 67.840% | 67.834% | 1.19916 | 78 | 2.89% | 46 / 28 | 62.16% | 58.97% / 10.00% | +18 | [+0.147, +1.148] pp |
| group_B_catboost_core | 67.618% | 67.612% | 1.19855 | 34 | 1.26% | 18 / 6 | 75.00% | 52.94% / 3.91% | +12 | [+0.111, +0.781] pp |
| group_B_logistic_core | 67.581% | 67.576% | 1.19842 | 18 | 0.67% | 12 / 1 | 92.31% | 66.67% / 2.61% | +11 | [+0.185, +0.670] pp |
| group_B_margin | 67.284% | 67.278% | 1.19905 | 11 | 0.41% | 5 / 2 | 71.43% | 45.45% / 1.09% | +3 | [-0.075, +0.333] pp |
| global_margin | 67.025% | 67.019% | 1.19919 | 52 | 1.93% | 15 / 19 | 44.12% | 28.85% / 3.26% | -4 | [-0.556, +0.259] pp |
| group_A_margin | 66.914% | 66.908% | 1.19977 | 78 | 2.89% | 28 / 35 | 44.44% | 35.90% / 6.09% | -7 | [-0.778, +0.223] pp |

### Fold stability

| Strategy | Fold | BA | Accuracy | Δ BA vs ORACLE02 fold |
|---|---:|---:|---:|---:|
| combined_logistic_core | 1 | 68.333% | 68.333% | +3.333 pp |
| combined_logistic_core | 2 | 71.481% | 71.481% | +4.815 pp |
| combined_logistic_core | 3 | 72.222% | 72.222% | +2.593 pp |
| combined_logistic_core | 4 | 72.778% | 72.778% | +7.593 pp |
| combined_logistic_core | 5 | 71.034% | 71.058% | +1.673 pp |
| global_logistic_core | 1 | 67.778% | 67.778% | +2.778 pp |
| global_logistic_core | 2 | 71.296% | 71.296% | +4.630 pp |
| global_logistic_core | 3 | 72.778% | 72.778% | +3.148 pp |
| global_logistic_core | 4 | 72.963% | 72.963% | +7.778 pp |
| global_logistic_core | 5 | 71.022% | 71.058% | +1.660 pp |
| global_catboost_core | 1 | 68.519% | 68.519% | +3.519 pp |
| global_catboost_core | 2 | 72.037% | 72.037% | +5.370 pp |
| global_catboost_core | 3 | 71.852% | 71.852% | +2.222 pp |
| global_catboost_core | 4 | 70.556% | 70.556% | +5.370 pp |
| global_catboost_core | 5 | 71.028% | 71.058% | +1.667 pp |
| global_catboost_core_stylometry | 1 | 68.704% | 68.704% | +3.704 pp |
| global_catboost_core_stylometry | 2 | 71.481% | 71.481% | +4.815 pp |
| global_catboost_core_stylometry | 3 | 72.222% | 72.222% | +2.593 pp |
| global_catboost_core_stylometry | 4 | 70.926% | 70.926% | +5.741 pp |
| global_catboost_core_stylometry | 5 | 70.102% | 70.130% | +0.741 pp |
| global_catboost_core_embedding | 1 | 67.778% | 67.778% | +2.778 pp |
| global_catboost_core_embedding | 2 | 71.667% | 71.667% | +5.000 pp |
| global_catboost_core_embedding | 3 | 70.926% | 70.926% | +1.296 pp |
| global_catboost_core_embedding | 4 | 70.926% | 70.926% | +5.741 pp |
| global_catboost_core_embedding | 5 | 72.139% | 72.171% | +2.778 pp |
| global_logistic_core_stylometry | 1 | 67.778% | 67.778% | +2.778 pp |
| global_logistic_core_stylometry | 2 | 71.852% | 71.852% | +5.185 pp |
| global_logistic_core_stylometry | 3 | 71.111% | 71.111% | +1.481 pp |
| global_logistic_core_stylometry | 4 | 70.556% | 70.556% | +5.370 pp |
| global_logistic_core_stylometry | 5 | 70.658% | 70.686% | +1.296 pp |
| combined_catboost_core | 1 | 69.074% | 69.074% | +4.074 pp |
| combined_catboost_core | 2 | 72.037% | 72.037% | +5.370 pp |
| combined_catboost_core | 3 | 71.481% | 71.481% | +1.852 pp |
| combined_catboost_core | 4 | 70.370% | 70.370% | +5.185 pp |
| combined_catboost_core | 5 | 68.621% | 68.646% | -0.741 pp |
| global_catboost_core_stylometry_embedding | 1 | 67.963% | 67.963% | +2.963 pp |
| global_catboost_core_stylometry_embedding | 2 | 70.556% | 70.556% | +3.889 pp |
| global_catboost_core_stylometry_embedding | 3 | 71.481% | 71.481% | +1.852 pp |
| global_catboost_core_stylometry_embedding | 4 | 70.556% | 70.556% | +5.370 pp |
| global_catboost_core_stylometry_embedding | 5 | 70.843% | 70.872% | +1.481 pp |
| global_logistic_core_embedding | 1 | 67.222% | 67.222% | +2.222 pp |
| global_logistic_core_embedding | 2 | 69.815% | 69.815% | +3.148 pp |
| global_logistic_core_embedding | 3 | 70.370% | 70.370% | +0.741 pp |
| global_logistic_core_embedding | 4 | 70.741% | 70.741% | +5.556 pp |
| global_logistic_core_embedding | 5 | 72.152% | 72.171% | +2.791 pp |
| global_logistic_core_stylometry_embedding | 1 | 67.407% | 67.407% | +2.407 pp |
| global_logistic_core_stylometry_embedding | 2 | 70.000% | 70.000% | +3.333 pp |
| global_logistic_core_stylometry_embedding | 3 | 68.889% | 68.889% | -0.741 pp |
| global_logistic_core_stylometry_embedding | 4 | 69.444% | 69.444% | +4.259 pp |
| global_logistic_core_stylometry_embedding | 5 | 72.337% | 72.356% | +2.976 pp |
| group_A_logistic_core | 1 | 66.481% | 66.481% | +1.481 pp |
| group_A_logistic_core | 2 | 67.963% | 67.963% | +1.296 pp |
| group_A_logistic_core | 3 | 70.556% | 70.556% | +0.926 pp |
| group_A_logistic_core | 4 | 67.037% | 67.037% | +1.852 pp |
| group_A_logistic_core | 5 | 69.176% | 69.202% | -0.185 pp |
| group_A_catboost_core | 1 | 66.111% | 66.111% | +1.111 pp |
| group_A_catboost_core | 2 | 67.407% | 67.407% | +0.741 pp |
| group_A_catboost_core | 3 | 70.185% | 70.185% | +0.556 pp |
| group_A_catboost_core | 4 | 65.926% | 65.926% | +0.741 pp |
| group_A_catboost_core | 5 | 69.547% | 69.573% | +0.185 pp |
| group_B_catboost_core | 1 | 65.741% | 65.741% | +0.741 pp |
| group_B_catboost_core | 2 | 66.852% | 66.852% | +0.185 pp |
| group_B_catboost_core | 3 | 69.630% | 69.630% | +0.000 pp |
| group_B_catboost_core | 4 | 65.926% | 65.926% | +0.741 pp |
| group_B_catboost_core | 5 | 69.923% | 69.944% | +0.562 pp |
| group_B_logistic_core | 1 | 65.370% | 65.370% | +0.370 pp |
| group_B_logistic_core | 2 | 66.667% | 66.667% | +0.000 pp |
| group_B_logistic_core | 3 | 70.000% | 70.000% | +0.370 pp |
| group_B_logistic_core | 4 | 65.741% | 65.741% | +0.556 pp |
| group_B_logistic_core | 5 | 70.109% | 70.130% | +0.747 pp |
| group_B_margin | 1 | 65.000% | 65.000% | +0.000 pp |
| group_B_margin | 2 | 66.667% | 66.667% | +0.000 pp |
| group_B_margin | 3 | 70.185% | 70.185% | +0.556 pp |
| group_B_margin | 4 | 65.185% | 65.185% | +0.000 pp |
| group_B_margin | 5 | 69.361% | 69.388% | +0.000 pp |
| global_margin | 1 | 65.000% | 65.000% | +0.000 pp |
| global_margin | 2 | 65.741% | 65.741% | -0.926 pp |
| global_margin | 3 | 69.630% | 69.630% | +0.000 pp |
| global_margin | 4 | 64.630% | 64.630% | -0.556 pp |
| global_margin | 5 | 70.102% | 70.130% | +0.741 pp |
| group_A_margin | 1 | 65.000% | 65.000% | +0.000 pp |
| group_A_margin | 2 | 66.481% | 66.481% | -0.185 pp |
| group_A_margin | 3 | 69.259% | 69.259% | -0.370 pp |
| group_A_margin | 4 | 64.815% | 64.815% | -0.370 pp |
| group_A_margin | 5 | 68.991% | 69.017% | -0.370 pp |

### Comparison with reference systems

| Strategy | Δ BA vs ORACLE02 | Δ BA vs ORACLE03 top-2 | Δ BA vs H09 | Δ BA vs EQUAL9 | Corrected/introduced vs ORACLE02 | vs ORACLE03 | vs H09 | vs EQUAL9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| combined_logistic_core | +4.001 pp | +3.668 pp | +5.668 pp | +5.779 pp | 223/115 | 220/121 | 284/131 | 286/130 |
| global_logistic_core | +4.001 pp | +3.668 pp | +5.668 pp | +5.778 pp | 229/121 | 239/140 | 288/135 | 290/134 |
| global_catboost_core | +3.632 pp | +3.299 pp | +5.298 pp | +5.409 pp | 196/98 | 204/115 | 263/120 | 264/118 |
| global_catboost_core_stylometry | +3.520 pp | +3.187 pp | +5.187 pp | +5.298 pp | 184/89 | 197/111 | 254/114 | 260/117 |
| global_catboost_core_embedding | +3.520 pp | +3.187 pp | +5.186 pp | +5.297 pp | 165/70 | 188/102 | 241/101 | 244/101 |
| global_logistic_core_stylometry | +3.223 pp | +2.890 pp | +4.890 pp | +5.001 pp | 191/104 | 209/131 | 258/126 | 266/131 |
| combined_catboost_core | +3.150 pp | +2.817 pp | +4.817 pp | +4.927 pp | 210/125 | 202/126 | 276/146 | 278/145 |
| global_catboost_core_stylometry_embedding | +3.111 pp | +2.778 pp | +4.778 pp | +4.889 pp | 180/96 | 190/115 | 247/118 | 248/116 |
| global_logistic_core_embedding | +2.891 pp | +2.558 pp | +4.557 pp | +4.668 pp | 147/69 | 191/122 | 228/105 | 228/102 |
| global_logistic_core_stylometry_embedding | +2.445 pp | +2.112 pp | +4.112 pp | +4.223 pp | 140/74 | 188/131 | 219/108 | 221/107 |
| group_A_logistic_core | +1.074 pp | +0.741 pp | +2.741 pp | +2.851 pp | 52/23 | 115/95 | 152/78 | 158/81 |
| group_A_catboost_core | +0.667 pp | +0.334 pp | +2.333 pp | +2.444 pp | 46/28 | 113/104 | 149/86 | 154/88 |
| group_B_catboost_core | +0.445 pp | +0.112 pp | +2.112 pp | +2.222 pp | 18/6 | 119/116 | 122/65 | 126/66 |
| group_B_logistic_core | +0.409 pp | +0.076 pp | +2.076 pp | +2.186 pp | 12/1 | 117/115 | 116/60 | 120/61 |
| group_B_margin | +0.111 pp | -0.222 pp | +1.778 pp | +1.889 pp | 5/2 | 117/123 | 109/61 | 113/62 |
| global_margin | -0.148 pp | -0.481 pp | +1.518 pp | +1.629 pp | 15/19 | 122/135 | 104/63 | 108/64 |
| group_A_margin | -0.259 pp | -0.592 pp | +1.407 pp | +1.518 pp | 28/35 | 111/127 | 117/79 | 130/89 |

## Feature utility and model selection

Feature importance is fold-model-specific and diagnostic. Core features include ORACLE02 top-1/top-2 labels and probabilities, margin/ratio/entropy, each checkpoint's probabilities and top-1 support for both candidate labels, family-level votes, and EQUAL9 agreement. The optional ablations append cached stylometry and frozen text embeddings. Top 20 features per fitted model are saved in `feature_importance.json`.

| Fitted model | Highest-ranked features |
|---|---|
| outer_1::global_logistic_core | N0027::p_top2, N0052::p_top2, N0009::p_top2, N0073::p_top2, N0049::p_top2, N0006::p_top2, N0009::p_top1, N0022::p_top2 |
| outer_1::global_logistic_core_stylometry | N0027::p_top2, N0052::p_top2, N0009::p_top2, N0073::p_top2, N0049::p_top2, N0006::p_top2, N0022::p_top2, N0009::p_top1 |
| outer_1::global_logistic_core_embedding | N0027::p_top2, N0052::p_top2, N0049::p_top2, N0009::p_top2, N0073::p_top2, N0057::p_top2, N0006::p_top2, top2_probability |
| outer_1::global_logistic_core_stylometry_embedding | N0027::p_top2, N0052::p_top2, N0049::p_top2, N0009::p_top2, N0073::p_top2, N0057::p_top2, N0006::p_top2, top2_probability |
| outer_1::global_catboost_core | top2_probability, N0027::p_top2, N0052::p_top2, N0001::p_top2, N0006::p_top2, N0005::p_top2, margin, top1_probability |
| outer_1::global_catboost_core_stylometry | top2_probability, N0027::p_top2, N0052::p_top2, margin, N0005::p_top2, N0009::p_top2, N0001::p_top2, N0056::p_top2 |
| outer_1::global_catboost_core_embedding | top2_probability, N0027::p_top2, N0052::p_top2, ratio, N0001::p_top2, N0005::p_top2, margin, N0006::p_top2 |
| outer_1::global_catboost_core_stylometry_embedding | top2_probability, margin, N0052::p_top2, N0027::p_top2, N0001::p_top2, N0009::p_top2, N0006::p_top2, N0003::p_top2 |
| outer_1::group_A_logistic_core | N0060::p_top2, N0073::votes_top2, oracle02_prob::claude-opus-5, N0019::votes_top2, N0009::p_top2, N0049::votes_top2, N0055::p_top1, N0055::votes_top1 |
| outer_1::group_A_catboost_core | N0009::p_top2, N0001::p_top2, N0032::p_top1, N0005::p_top2, oracle02_prob::qwen3.8-max, N0010::p_top2, oracle02_prob::claude-opus-5, N0030::p_top2 |
| outer_1::group_B_logistic_core | N0031::p_top2, N0014::votes_top2, N0054::votes_top2, N0072::votes_top2, N0037::votes_top2, N0070::votes_top2, N0009::votes_top2, N0039::votes_top1 |
| outer_1::group_B_catboost_core | N0031::p_top2, N0020::p_top2, N0023::p_top2, N0009::p_top2, N0034::p_top1, N0019::p_top2, N0021::p_top2, N0006::p_top2 |
| outer_2::global_logistic_core | N0027::p_top2, N0052::p_top2, N0009::p_top2, N0073::p_top2, N0049::p_top2, N0052::p_top1, N0027::p_top1, N0022::p_top2 |
| outer_2::global_logistic_core_stylometry | N0027::p_top2, N0052::p_top2, N0073::p_top2, N0009::p_top2, N0049::p_top2, N0007::p_top2, N0052::p_top1, N0027::p_top1 |
| outer_2::global_logistic_core_embedding | N0027::p_top2, N0052::p_top2, N0049::p_top2, N0009::p_top2, N0073::p_top2, N0005::p_top2, N0007::p_top2, N0022::p_top2 |
| outer_2::global_logistic_core_stylometry_embedding | N0052::p_top2, N0027::p_top2, N0049::p_top2, N0073::p_top2, N0009::p_top2, N0007::p_top2, top1::gemini-3.8-flash, N0052::p_top1 |
| outer_2::global_catboost_core | ratio, N0052::p_top2, N0009::p_top2, top2_probability, N0005::p_top2, N0027::p_top2, margin, N0049::p_top2 |
| outer_2::global_catboost_core_stylometry | top2_probability, ratio, N0052::p_top2, N0009::p_top2, N0027::p_top2, N0081::p_top2, N0006::p_top2, N0005::p_top2 |
| outer_2::global_catboost_core_embedding | ratio, top2_probability, margin, N0052::p_top2, N0005::p_top2, N0009::p_top2, N0027::p_top2, N0006::p_top2 |
| outer_2::global_catboost_core_stylometry_embedding | top2_probability, margin, N0052::p_top2, ratio, N0009::p_top2, N0005::p_top2, N0003::p_top2, N0049::p_top2 |
| outer_2::group_A_logistic_core | N0073::votes_top2, N0022::votes_top1, N0055::votes_top1, N0106::votes_top1, N0077::votes_top1, N0009::p_top2, N0049::votes_top2, N0022::votes_top2 |
| outer_2::group_A_catboost_core | N0009::p_top2, N0001::p_top2, top2_probability, N0058::p_top1, N0030::p_top2, N0074::p_top1, oracle02_prob::claude-fable-5, N0055::p_top1 |
| outer_2::group_B_logistic_core | N0065::p_top1, N0039::votes_top2, N0054::votes_top2, N0035::votes_top2, N0059::p_top2, N0036::votes_top1, N0077::votes_top1, N0063::votes_top2 |
| outer_2::group_B_catboost_core | N0009::p_top2, N0036::p_top1, N0065::p_top1, N0004::p_top2, oracle02_prob::glm-5.3, N0076::p_top2, N0012::p_top2, N0002::p_top2 |
| outer_3::global_logistic_core | N0027::p_top2, N0052::p_top2, N0009::p_top2, N0073::p_top2, N0049::p_top2, N0027::p_top1, N0006::p_top2, N0081::p_top2 |
| outer_3::global_logistic_core_stylometry | N0027::p_top2, N0052::p_top2, N0073::p_top2, N0009::p_top2, N0049::p_top2, N0081::p_top2, N0006::p_top2, N0027::p_top1 |
| outer_3::global_logistic_core_embedding | N0027::p_top2, N0052::p_top2, N0009::p_top2, N0049::p_top2, N0073::p_top2, N0006::p_top2, N0005::p_top2, top1::gemini-3.8-flash |
| outer_3::global_logistic_core_stylometry_embedding | N0052::p_top2, N0027::p_top2, N0049::p_top2, N0073::p_top2, N0009::p_top2, N0006::p_top2, top1::gemini-3.8-flash, N0073::votes_top2 |
| outer_3::global_catboost_core | N0052::p_top2, N0009::p_top2, ratio, margin, N0005::p_top2, N0027::p_top2, N0051::p_top2, top2_probability |
| outer_3::global_catboost_core_stylometry | N0052::p_top2, ratio, N0009::p_top2, top2_probability, N0001::p_top2, N0027::p_top2, N0009::votes_top2, N0005::p_top2 |
| outer_3::global_catboost_core_embedding | N0052::p_top2, ratio, N0009::p_top2, margin, top2_probability, N0027::p_top2, N0005::p_top2, N0009::votes_top2 |
| outer_3::global_catboost_core_stylometry_embedding | N0052::p_top2, ratio, top2_probability, N0001::p_top2, N0009::p_top2, N0005::p_top2, top1::gemini-3.8-flash, N0027::p_top2 |
| outer_3::group_A_logistic_core | N0055::votes_top1, N0077::votes_top1, N0073::votes_top2, N0055::p_top1, N0049::votes_top2, N0030::p_top2, N0027::votes_top2, N0030::p_top1 |
| outer_3::group_A_catboost_core | N0009::p_top2, N0055::p_top1, N0030::p_top2, N0022::p_top2, N0081::p_top1, N0055::votes_top1, N0001::p_top2, N0040::p_top1 |
| outer_3::group_B_logistic_core | N0035::votes_top2, N0040::p_top1, N0077::p_top1, N0006::votes_top2, N0032::p_top2, N0048::votes_top1, N0009::p_top1, N0014::votes_top2 |
| outer_3::group_B_catboost_core | N0009::p_top2, N0005::p_top2, N0004::p_top2, N0001::p_top2, N0058::p_top1, N0034::p_top2, N0006::p_top2, N0081::p_top2 |
| outer_4::global_logistic_core | N0027::p_top2, N0052::p_top2, N0009::p_top2, N0006::p_top2, N0073::p_top2, N0049::p_top2, N0027::p_top1, N0073::p_top1 |
| outer_4::global_logistic_core_stylometry | N0027::p_top2, N0052::p_top2, N0009::p_top2, N0006::p_top2, N0073::p_top2, N0049::p_top2, N0027::p_top1, N0073::p_top1 |
| outer_4::global_logistic_core_embedding | N0027::p_top2, N0052::p_top2, N0049::p_top2, N0009::p_top2, N0073::p_top1, N0073::p_top2, N0006::p_top2, N0057::p_top2 |
| outer_4::global_logistic_core_stylometry_embedding | N0027::p_top2, N0052::p_top2, N0049::p_top2, N0009::p_top2, N0073::p_top2, N0006::p_top2, N0073::p_top1, N0057::p_top2 |
| outer_4::global_catboost_core | N0052::p_top2, N0009::p_top2, margin, top2_probability, ratio, N0001::p_top2, N0027::p_top2, N0049::p_top2 |
| outer_4::global_catboost_core_stylometry | ratio, N0052::p_top2, margin, top2_probability, N0001::p_top2, N0027::p_top2, N0009::p_top2, N0005::p_top2 |
| outer_4::global_catboost_core_embedding | N0052::p_top2, top2_probability, ratio, margin, N0009::p_top2, N0005::p_top2, N0001::p_top2, N0049::p_top2 |
| outer_4::global_catboost_core_stylometry_embedding | margin, N0052::p_top2, N0001::p_top2, N0009::p_top2, ratio, top2_probability, N0027::p_top2, N0005::p_top2 |
| outer_4::group_A_logistic_core | N0106::votes_top1, N0015::votes_top2, N0078::p_top1, N0059::votes_top1, N0009::p_top2, N0073::votes_top2, N0060::p_top2, N0035::p_top1 |
| outer_4::group_A_catboost_core | N0009::p_top2, N0055::p_top1, N0030::p_top2, N0106::p_top1, N0047::p_top2, N0001::p_top2, N0022::p_top1, N0006::p_top2 |
| outer_4::group_B_logistic_core | N0039::votes_top2, N0056::votes_top2, N0077::votes_top1, N0003::votes_top2, N0009::votes_top2, N0009::p_top2, N0070::votes_top2, N0038::votes_top1 |
| outer_4::group_B_catboost_core | N0009::p_top2, N0006::p_top2, N0005::p_top2, family::catboost::votes_top1, N0056::p_top1, N0026::p_top2, N0039::p_top2, N0006::p_top1 |
| outer_5::global_logistic_core | N0027::p_top2, N0052::p_top2, N0009::p_top2, N0049::p_top2, N0073::p_top2, N0006::p_top2, N0030::p_top2, N0027::p_top1 |
| outer_5::global_logistic_core_stylometry | N0027::p_top2, N0052::p_top2, N0009::p_top2, N0049::p_top2, N0073::p_top2, N0006::p_top2, N0030::p_top2, N0027::p_top1 |
| outer_5::global_logistic_core_embedding | N0027::p_top2, N0052::p_top2, N0009::p_top2, N0049::p_top2, N0073::p_top2, N0006::p_top2, N0030::p_top2, N0052::p_top1 |
| outer_5::global_logistic_core_stylometry_embedding | N0027::p_top2, N0009::p_top2, N0052::p_top2, N0049::p_top2, N0073::p_top2, N0006::p_top2, N0081::p_top2, N0030::p_top2 |
| outer_5::global_catboost_core | N0027::p_top2, N0052::p_top2, top2_probability, N0009::p_top2, N0001::p_top2, ratio, N0006::p_top2, N0049::p_top2 |
| outer_5::global_catboost_core_stylometry | N0052::p_top2, margin, top2_probability, N0027::p_top2, N0001::p_top2, ratio, N0009::p_top2, N0006::p_top2 |
| outer_5::global_catboost_core_embedding | N0052::p_top2, top2_probability, ratio, margin, N0027::p_top2, N0005::p_top2, N0009::p_top2, N0001::p_top2 |
| outer_5::global_catboost_core_stylometry_embedding | N0052::p_top2, ratio, top2_probability, margin, N0027::p_top2, N0006::p_top2, N0001::p_top2, N0005::p_top2 |
| outer_5::group_A_logistic_core | N0030::p_top2, N0015::votes_top2, N0019::votes_top2, N0022::votes_top1, N0015::votes_top1, N0027::votes_top2, N0049::votes_top2, N0009::p_top2 |
| outer_5::group_A_catboost_core | N0009::p_top2, N0030::p_top2, oracle02_prob::claude-fable-5, N0001::p_top2, N0027::p_top2, N0022::p_top1, N0078::p_top2, N0005::p_top2 |
| outer_5::group_B_logistic_core | N0039::votes_top1, N0039::votes_top2, N0036::votes_top1, N0036::p_top1, N0070::votes_top2, N0048::votes_top1, N0054::votes_top2, N0077::p_top1 |
| outer_5::group_B_catboost_core | N0037::p_top1, N0036::p_top1, N0071::p_top1, N0006::p_top2, N0004::p_top2, N0001::p_top2, N0002::p_top2, N0055::p_top2 |

## Per-label impact

Per-label precision and recall for the top-ranked strategy (**combined_logistic_core**) versus ORACLE02:

| Label | ORACLE02 precision | Best precision | Δ | ORACLE02 recall | Best recall | Δ |
|---|---:|---:|---:|---:|---:|---:|
| claude-fable-5 | 42.22% | 48.81% | +6.59 pp | 51.01% | 55.03% | +4.03 pp |
| claude-opus-4-6 | 84.67% | 84.77% | +0.10 pp | 84.67% | 85.33% | +0.67 pp |
| claude-opus-4-8 | 78.77% | 79.73% | +0.96 pp | 76.67% | 78.67% | +2.00 pp |
| claude-opus-5 | 88.72% | 89.55% | +0.83 pp | 78.67% | 80.00% | +1.33 pp |
| claude-sonnet-5 | 70.51% | 73.33% | +2.82 pp | 73.33% | 73.33% | +0.00 pp |
| gemini-3.1-pro-preview | 92.54% | 90.78% | -1.76 pp | 82.67% | 85.33% | +2.67 pp |
| gemini-3.6-flash | 68.82% | 73.05% | +4.24 pp | 85.33% | 81.33% | -4.00 pp |
| gemini-3.7-flash | 62.96% | 62.42% | -0.54 pp | 22.67% | 65.33% | +42.67 pp |
| gemini-3.8-flash | 52.16% | 67.61% | +15.45 pp | 80.67% | 64.00% | -16.67 pp |
| glm-5.3 | 42.14% | 44.38% | +2.24 pp | 44.67% | 50.00% | +5.33 pp |
| glm-5.3-flash | 55.21% | 56.69% | +1.47 pp | 60.00% | 59.33% | -0.67 pp |
| gpt-5.5 | 64.63% | 68.87% | +4.25 pp | 63.33% | 69.33% | +6.00 pp |
| gpt-5.6-luna | 69.29% | 73.79% | +4.51 pp | 64.67% | 71.33% | +6.67 pp |
| gpt-5.6-sol | 64.75% | 67.38% | +2.62 pp | 52.67% | 63.33% | +10.67 pp |
| gpt-5.6-terra | 59.12% | 69.68% | +10.56 pp | 71.33% | 72.00% | +0.67 pp |
| gpt-6-astra | 87.94% | 87.92% | -0.02 pp | 82.67% | 87.33% | +4.67 pp |
| kimi-k3 | 70.00% | 70.25% | +0.25 pp | 51.33% | 56.67% | +5.33 pp |
| qwen3.8-max | 75.15% | 81.70% | +6.55 pp | 82.67% | 83.33% | +0.67 pp |

Full precision, recall, F1, TP, FP, FN and support for every baseline and variant are in `per_label_metrics.csv`. The row-level predictions preserve V0 IDs and prompt IDs in `row_predictions.parquet`.

## Conclusions

1. **Swap detection:** best approach combined_logistic_core triggers 404 swaps; its held-out top-2-correct target precision/recall is 55.2%/48.5%.
2. **Useful features:** global logistic core is most consistently led by checkpoint probabilities assigned to top-2 (notably N0027::p_top2, N0052::p_top2, N0009::p_top2, N0073::p_top2, N0049::p_top2); adding stylometry or embeddings did not improve over the core global logistic result.
3. **Targeted groups:** Group A logistic improved BA over ORACLE02 in 4/5 folds (52/23 corrected/introduced); Group A CatBoost in 5/5. Group B logistic improved in 4/5 (12/1); Group B CatBoost in 4/5. Their target recall remains low.
4. **Global and combined:** global logistic reached 71.168% BA; combined logistic reached 71.169%. The combined gain over global is +0.000 pp, effectively tied.
5. **Oracle headroom:** top-two oracle BA is 84.213%; combined_logistic_core gains 4.001 pp of the 17.045 pp available over ORACLE02 (23.5% recovered).
6. **Against ORACLE03:** combined_logistic_core changes BA by +3.668 pp versus the 67.500% top-2 specialist. Best-strategy prompt-cluster bootstrap intervals are reported in the next table; these remain V0 diagnostics.
7. **Kaggle readiness:** do not submit on this evidence alone. V0 influenced checkpoint, roster and prior ORACLE02 selection; no independent Kaggle score is estimated.

### Bootstrap interval for the selected best strategy

Selection of the best strategy used the pooled cross-fitted V0 scores, so these paired bootstrap intervals are descriptive after selection, not an unbiased model-selection interval.

| Reference | Best strategy Δ BA, 95% prompt-group bootstrap CI |
|---|---:|
| ORACLE02 | [+2.706, +5.371] pp |
| ORACLE03_top2 | [+2.370, +5.005] pp |
| H09 | [+4.295, +7.149] pp |
| EQUAL9 | [+4.332, +7.259] pp |

## Three recommendations

1. Keep combined_logistic_core as the strongest development candidate: 223 corrected versus 115 introduced against ORACLE02; it exceeds ORACLE03 by +3.668 pp on these folds.
2. Prefer the simpler global logistic model for the next controlled comparison; combined routing adds only a negligible score difference and adds specialist thresholds.
3. Require an unseen labeled evaluation or Kaggle submission before changing the 0.581 public-score reference.

## Limitations and artifacts

V0 is reused for base checkpoint and trust-list selection. The outer-fold swap-model predictions are held out by prompt, but the saved ORACLE02 held-out probabilities come from its prior outer-fold workflow, whose model/hyperparameter choices were also made using the outer training rows. To keep swap-model training leakage controlled, ORACLE02 training examples receive additional nested predictions generated exclusively from other prompt groups within that outer training partition. For speed, nested ORACLE02 fits use the modal existing CatBoost configuration (depth 3, 120 trees, L2 10, rsm .25) and fixed 50:50 EQUAL9 blending; this locked setting is recorded, not retuned on the nested held-out labels. The independent Kaggle generalization remains unknown.
Artifacts: `metrics.json`, `row_predictions.parquet`, `per_label_metrics.csv`, `fold_metrics.csv`, `fold_decisions.csv`, `feature_importance.json`, `nested_baseline_outer_*.npz/.json`, `features_outer_*.npz`, and fold-specific model files under `models/`.
