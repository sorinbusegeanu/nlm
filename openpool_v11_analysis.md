# Name That LLM Open Pool V11: exploratory analysis

**Scope.** Read-only analysis of `data/raw/open_pool/` and the local official reference `data/raw/train.csv`. No training, relabeling, or source-data edits were performed. Supporting tables and figures are in [`analysis-output/openpool_v11/`](analysis-output/openpool_v11/). The analysis helper is [`analyze_openpool.py`](analysis-output/openpool_v11/analyze_openpool.py), so measurements can be reproduced with `/home/zodrak/cuda13.x/bin/python analysis-output/openpool_v11/analyze_openpool.py`.

## 1. Executive summary

The main Open Pool contains **190,588 passages across 18 labels, 55 source datasets, and 73 source files**. It is sizeable, but its rows are not independent: a row has a mean 4.62 `n_sources` (median 3), and source identity is strongly associated with the assigned label (Cramér’s V **0.336**). A classifier may learn collection/source artifacts as well as model style. This is the largest evaluation risk visible in the files.

Label counts are unbalanced by **6.57×** (21,550 Gemini 3.8 Flash vs 3,278 Gemini 3.7 Flash). Recovered prompt coverage is partial: `prompts.parquet` has text for 13,021 prompt keys and reconciles to 149,022 pool rows (78.2%); the remaining 41,566 rows have only prompt IDs/evidence or no prompt. Of the recovered prompts, 592 (4.55%) contain all 18 labels. The supplied `panel_gaps.csv` covers only the 4,472 prompts with 10–17 labels; it excludes 7,957 prompts with 1–9 labels.

The available files cannot establish the correctness of every row’s model label. They do provide useful provenance and cleaning summaries, but no row-level generation logs or independently authenticated labels. In particular, 10,593 Gemini 3.7 rows are separated into an explicitly unverified sidecar; the 3,278 retained Gemini 3.7 rows have no populated `origin_hint` and come from only four source datasets. Treat this class as route-uncertain, not verified ground truth.

The publisher reports that cleaning improved balanced accuracy by roughly 2.6 points in a two-seed 45k-row comparison. The supplied round-1 scores independently yield +2.74 points for cleaned C5 over raw R5. Round-4 paired estimates favor typography matching (+2.63 pp; published interval +1.19 to +3.93 pp) and Vertex-only Gemini 3.7 (+0.93 pp; interval 0 to +1.86 pp), but those paired point estimates do not reconcile with the simple two-seed run table. There are only two seeds and the same small official train set is reused, so evidence should be considered promising but not conclusive.

Marginal `match_weight` reweighting closely matches many measured style-marker rates to the official train distribution, as expected from its construction. It cannot correct source/prompt confounding or match the full joint text distribution. A lexical nearest-neighbor scan found no train/Open Pool pair with cosine similarity >=0.8 under the stated TF-IDF method, but this is not a semantic or prompt-overlap test.

**Overall assessment:** useful as a broad, exploratory auxiliary corpus and for controlled ablations; unsuitable as unqualified row-level ground truth or as the sole basis for model-selection claims. Use prompt-grouped and source-grouped evaluation, report class-balanced metrics, and exclude/flag route-uncertain labels.

## 2. File inventory and dataset structure

The complete observed inventory, dimensions, columns, byte sizes, and purpose are in [`file_inventory.csv`](analysis-output/openpool_v11/tables/file_inventory.csv). The five supplied PNGs were opened and read as publisher figures. Key structures:

| File | Dimensions | Role and relationship |
|---|---:|---|
| `ntl_open_pool.parquet` | 190,588 × 23 | Main retained passage table; model/provider, source/license, prompt, similarity, flags, and weight fields. |
| `prompts.parquet` | 13,021 × 4 | Recovered prompt text indexed by `prompt_canon`; its passage counts reconcile exactly to main-pool rows for the keys it contains. |
| `gemini37_unverified.parquet` | 10,593 × 19 | Gemini 3.7 route-uncertain sidecar; its IDs do not overlap the retained pool. |
| `panel_gaps.csv` | 4,472 × 4 | Missing-label lists for prompts with 10–17 represented labels, not every incomplete prompt. |
| `per_model.csv` | 18 × 5 | Publisher per-label counts and summary fields; reconciled against the pool in the output tables. |
| `sources.csv` | 55 × 4 | Publisher source-level counts/summary; reconciled against measured source rows. |
| `typography_match.csv` | 252 × 6 | Publisher marker rates by model before and after weighting. |
| `style_gap.csv` | 9 × 6 | Aggregate style gaps for older V7/V9 comparisons; it is not a direct V11 measurement. |
| `dataq4_paired.csv` | 20 × 8 | Publisher paired deltas and 95% intervals for round-4 variants/metrics. |
| `dataq4_runs.csv` | 10 × 9 | Round-4 scores for five variants across two seeds. |
| `dataq_results.csv` | 18 × 9 | Round-1/2 variant scores, two seeds per variant. |
| `removed_rows.csv` | 88,508 × 2 | Removed IDs and reasons only; passage content is absent, so removals cannot be independently re-evaluated. |
| `removal_summary.csv` | 12 × 2 | Cumulative removal-reason accounting, including the retained V11 count. |
| `v11_append_log.csv` | 16 × 2 | Publisher’s accepted/filtered row accounting for the V11 append. |
| `fig1_clean_vs_raw.png`–`fig5_v11_overview.png` | 5 PNGs | Publisher diagrams/results, summarized in sections 4, 5, and 8. |

The main table has no date/timestamp/version columns, so generation-date or precise model-version distributions cannot be derived. All 190,588 rows have a `source_license` value: 158,393 CC BY 4.0, 28,154 CC-BY-SA-4.0, and 4,041 CC0-1.0. Licensing metadata is not evidence of label correctness.

Text lengths were measured from text using a Unicode word regex, a sentence-boundary heuristic, character counts, and `cl100k_base` token counts. These are measurement conventions, not the competition’s tokenizer or guaranteed sentence counts. Per-class means, medians, and tails are in [`text_style_by_model.csv`](analysis-output/openpool_v11/tables/text_style_by_model.csv); corpus summaries are in [`corpus_text_style_summary.csv`](analysis-output/openpool_v11/tables/corpus_text_style_summary.csv). The sampled language audit classified all 4,500 stratified Open Pool samples and all 2,699 official rows as English; this says nothing about rare languages outside the sample. See [`language_sample_counts.csv`](analysis-output/openpool_v11/tables/language_sample_counts.csv).

## 3. Label and source distributions

The pool is concentrated in Google (51,134 rows), OpenAI (64,029), and Anthropic (38,724), with smaller ZAI (23,004), Moonshot (9,426), and Qwen (4,271) groups. The exact per-label counts, provider labels, proportions, official-train row counts, and prompt coverage are in [`label_distribution_and_coverage.csv`](analysis-output/openpool_v11/tables/label_distribution_and_coverage.csv). Counts range from 3,278 Gemini 3.7 Flash (1.72%) and 4,271 Qwen3.8-max (2.24%) to 21,550 Gemini 3.8 Flash (11.31%).

There are 55 `source_dataset` values and 73 `source_file` values. Source and label association has Cramér’s V **0.336** (chi-square 366,778; nominal p underflows to 0 at machine precision). This is a substantial association, not a label-validation statistic. `lab` and `model` have Cramér’s V 1.0 because provider is determined by the label naming scheme; this is expected and not independent evidence.

The main pool has **zero** duplicate rows under NFKC normalization, case-folding, and whitespace collapsing, and no normalized text with conflicting labels. There are no exact normalized-text matches between the pool and official `train.csv`. The pool and official train use different prompt identifiers, so prompt IDs alone cannot establish prompt overlap. These duplicate checks are recorded in [`duplicate_summary.csv`](analysis-output/openpool_v11/tables/duplicate_summary.csv).

## 4. Label reliability assessment

### Observable facts

- Main-pool row IDs are disjoint from the 10,593-row Gemini 3.7 unverified sidecar; removed-row IDs are also disjoint from retained IDs.
- `origin_hint` is populated on only 33,093 retained rows (17.4%); it is empty for every retained Gemini 3.7 row.
- Retained Gemini 3.7 has 3,278 rows, only four source datasets and five source files. One Vertex-named source contributes 2,007 rows (61.2%). The sidecar has 10,593 rows across eight source datasets and nine files.
- The main dataset records multiple mirrored/source memberships per passage (`n_sources` mean 4.62). Those are not independent label votes.
- Publisher removal accounting sums to 88,508 historical removals. Largest reasons are `off_genre_source` 30,992, `api_dump` 13,114, `gemini37_route_unverified` 10,593, and `source_unavailable` 9,815. `removed_rows.csv` lacks passage text, limiting independent scrutiny.
- The V11 append ledger arithmetic is internally consistent: 27,618 candidate rows less documented duplicate, format, refusal, truncation, typography, and train-similarity filters yields 10,294 appended rows; 10,593 Gemini 3.7 rows were moved aside.

### Author claims versus independent conclusions

Figure 5 attributes the 3.7/3.8 confusion to some non-Vertex endpoints returning 3.8 for 3.7 requests after 2026-10-08 and reports source-level AUCs mostly near 0.5. The supplied artifacts do not contain the request/response logs or source-level AUC data needed to independently verify that claim. The observed sidecar/main style and source differences are descriptive only; they cannot identify which label is correct.

No source can be certified “trustworthy” from this bundle alone. Greater traceability is suggested by explicit route/source names and consistent provenance fields; lower traceability is suggested by empty origin metadata and a disputed route, especially the Gemini 3.7 group. Two single-label source datasets (`tadeuhayashida/ntl-claude-fable-5-generations`, n=1,609; `prvsiyan/prvsiyans-extended-name-that-llm-dataset`, n=266) are not thereby verified: purity can indicate a curated source or a shortcut/leakage risk. Source-level measurements are available in [`source_quality_signals.csv`](analysis-output/openpool_v11/tables/source_quality_signals.csv) and [`source_model_mix.csv`](analysis-output/openpool_v11/tables/source_model_mix.csv).

### What the five supplied figures say

- **Figure 1 (clean vs raw):** reports 45,000 passages per run and about +0.026 balanced accuracy for cleaned V1–5 vs raw V1–5, using Qwen3-1.7B LoRA and two seeds. Recalculation from `dataq_results.csv` gives +0.0274.
- **Figure 2 (label gain):** reports that labels with more junk/off-genre material benefited more from cleaning (Spearman ρ=0.64 across 18 labels). The per-label junk-removal quantities needed to reproduce it are absent, so this is an author-reported association.
- **Figure 3 (style gap):** shows some style gaps narrowing after cleaning and a persistent first-person-plural (“we/our/us”) difference attributed by the author to host prompts. The plotted `style_gap.csv` data are V7/V9-era, not a V11 pool comparison.
- **Figure 4 (round 2):** reports little benefit from stricter filters, approximately −0.001 after removing around 70k Kaggle-proxy rows, and approximately −0.004 for keeping the 70% most train-like rows, with weak-label recall −0.016. The latter score direction is also visible in the supplied two-seed round-2 CSV; the proxy-row/recall figures are not fully reconstructible there.
- **Figure 5 (V11 overview):** gives the publisher’s ledger 198,507 V10 rows −7,620 unavailable-source rows +10,294 append rows −10,593 route-uncertain Gemini 3.7 rows = 190,588 main V11 rows. It reports 13,021 recovered prompt texts, 592 complete 18-label panels, and 4,472 panels with 10–17 labels; it omits the 7,957 recovered prompts with 1–9 labels from the `panel_gaps.csv` view. It also reports source-level Gemini 3.7-vs-3.8 AUCs mostly near chance (0.465–0.566, with a 0.776 3.6-vs-3.8 result in a Harshit source). The source AUC measurements and raw predictions are not included, so they remain unverified claims.

## 5. Prompt coverage analysis

The 13,021 `prompts.parquet` keys reconcile exactly with the number of retained passages and distinct labels in the main pool for each recovered key; their sum is **149,022 rows**. The other **41,566** rows have prompt evidence but not recovered prompt text: `id_only` 39,641 and `none` 1,925. `prompt_canon` has 31,355 unique values in the main pool; 1,925 rows have it blank. Evidence counts are `own_bank` 75,924, `own_text` 31,516, `other_text` 27,474, `centroid` 14,108, `id_only` 39,641, and `none` 1,925. A recovered prompt should therefore not be assumed equally direct or certain across evidence categories.

Among recovered prompts, 592 (4.55%) have all 18 labels and 12,429 are incomplete. 7,957 have only 1–9 labels; the remaining 4,472 have 10–17 labels and are exactly the population listed in `panel_gaps.csv`. The lists match the observed missing labels for all 4,472 supplied records. They do **not** describe the 1–9-label prompts, despite the broad figure wording. Full distributions are in [`prompt_panel_distribution.csv`](analysis-output/openpool_v11/tables/prompt_panel_distribution.csv) and the audit is in [`panel_gap_audit.csv`](analysis-output/openpool_v11/tables/panel_gap_audit.csv).

Among recovered prompts, the strongest label coverage is Gemini 3.8 Flash (8,627 prompts, 66.3%), GPT-5.6 Luna (7,894, 60.6%), and GPT-5.6 Terra (7,723, 59.3%). Weakest is Gemini 3.7 Flash (3,112, 23.9%), followed by Qwen3.8-max (3,481, 26.7%), Claude Opus 5 (4,309, 33.1%), Kimi K3 (4,312, 33.1%), and Gemini 3.1 Pro (4,477, 34.4%). This is measured only over prompt texts that were recovered; missing prompt text may have different coverage.

Complete panels make within-prompt comparisons possible and reduce prompt-content confounding. However, a classifier evaluated on passage rows can still benefit from prompt-specific response content, and complete prompts represent only 27,891 rows (14.6% of the pool). Prompt-balanced, group-held-out reporting is needed; complete-panel-only scoring is a useful complementary metric, not a substitute for representative coverage.

## 6. Style and typography analysis

The analysis measured passage length, sentence and paragraph heuristics, type-token ratio, capitalization, function-word rates, and punctuation/formatting markers. Per-label results are in [`text_style_by_model.csv`](analysis-output/openpool_v11/tables/text_style_by_model.csv); feature-wise normalized style patterns are visualized in [`04_style_marker_zscores.svg`](analysis-output/openpool_v11/figures/04_style_marker_zscores.svg). Many markers are associated with label in row-level tests, with Cramér’s V ranging from 0.03 for first-person marker presence to 0.64 for em-dash presence. These associations are not causal or source-adjusted. Source and prompt differences can create or amplify them.

Length differences are statistically detectable with this large sample, but small in effect size: Kruskal–Wallis epsilon-squared is about 0.005 for characters, 0.010 for regex word count, 0.009 for `cl100k_base` token count, and 0.035 for type-token ratio. Nominal p-values are tiny, but the tests treat correlated rows as independent and do not adjust for source/prompt clustering. Punctuation marker rates and counts-per-100-words are both reported; binary rate says whether a passage uses a marker at least once, while frequency captures repeated use.

Compared with official train, raw Open Pool differs on several marginal marker rates after equal-label averaging: digit-containing passage rate is +11.43 percentage points, word-hyphen rate +10.93 pp, first-person plural (“we/our/us”) rate −14.29 pp, and semicolon rate −7.86 pp. Reweighting by publisher `match_weight` reduces these measured marginal gaps to roughly 0.01–0.81 pp for those examples; all marker comparisons are in [`official_vs_openpool_style.csv`](analysis-output/openpool_v11/tables/official_vs_openpool_style.csv).

This explains the reported typography-matching improvement plausibly: style markers differ between the pool and official train, and the weights explicitly reduce those measured differences. It does not prove that matching these markers caused the gain; weights may also alter source/model composition, and marginal marker matching does not control the prompt or source mix. Figure 3’s persistent “we/our/us” gap is consistent with the measured feature’s known prompt/source sensitivity, but the provided `style_gap.csv` describes V7/V9, not current V11.

`match_weight` mean is 0.99999, median 0.821, p90 1.934, p95 2.583, p99 4.315, p99.9 7.166, max 50. Only 0.61% exceed 5, 0.059% exceed 10, and 0.0005% are at the cap. Overall effective sample size (ESS) is 104,796 (55.0% of nominal 190,588). Per-model ESS and weight summaries are in [`text_style_by_model.csv`](analysis-output/openpool_v11/tables/text_style_by_model.csv); per-label ESS loss means effective information is materially below the raw row count.

Limitations of the published matching approach: the available table demonstrates matching only for listed marginal style markers; it does not document a full joint-distribution objective, uncertainty, source/prompt covariate balance, or sensitivity to clipping. Matching marker rates can still leave vocabulary, semantics, formatting interactions, and provenance shortcuts unbalanced. It also cannot fix mislabeled rows.

## 7. Comparison with official Kaggle data

Local `train.csv` has **2,699 passages**, 18 labels, and 150 prompt IDs. Seventeen labels have 150 rows; one has 149. Each prompt has 17 or 18 rows. This is far more class-balanced than the Open Pool. Train prompts are IDs (`p000` etc.), while Open Pool uses canonical hashes and recovered text; there is no provided join table to determine exact prompt-text overlap. The absence of exact normalized-text duplicates is not evidence that prompts are disjoint.

Open Pool passages are generally longer than the official set on some labels and shorter on others; detailed comparable metrics are in [`corpus_text_style_summary.csv`](analysis-output/openpool_v11/tables/corpus_text_style_summary.csv). There are no exact NFKC/case/whitespace-normalized text matches. A full-pool lexical screen used English-stopword TF-IDF over word unigrams/bigrams (`min_df=2`, max 150k features) and, for each official passage, its nearest Open Pool passage. Similarity quantiles: median 0.197, p90 0.276, p95 0.294, p99 0.335, maximum 0.388; none reached 0.8. See [`near_duplicate_summary.json`](analysis-output/openpool_v11/near_duplicate_summary.json) and [`official_to_openpool_nearest_text.csv`](analysis-output/openpool_v11/tables/official_to_openpool_nearest_text.csv). This is a lexical passage screen, not a semantic or prompt overlap assessment.

The main pool records `max_cos_train` median 0.09 (max 0.30) and `max_sim_train_prompt` median 0.084 (max 0.399); append notes say candidate rows beyond similarity thresholds were filtered. These are publisher-computed metadata, not independently reconstructed from their undisclosed embedding/model/method. The independent lexical screen is a different metric and must not be conflated with these fields.

Thus Open Pool is similar to the official task in having the same 18 named model labels and short-response format, but differs substantially in class balance, sources, prompt coverage, and style distribution. There is not enough information to quantify prompt distribution overlap or source shifts directly. Any validation on official train rows is small and potentially sensitive to repeated experiment selection.

## 8. Analysis of the author’s experiments

These are publisher-provided experiments, independently recalculated from their CSVs where possible. No model was trained for this analysis.

### Round 1 and 2

Each run uses 45,000 passages and two seeds. Round 1 mean balanced accuracy: raw R5 (v1–5) **0.4657**, hard-junk-filtered H5 **0.4744**, cleaned C5 **0.4931**, raw R7 **0.4644**, cleaned C7 **0.4920**. C5 exceeds R5 by +0.0273 and +0.0275 on the two seeds (mean +0.0274); C7 exceeds R7 by +0.0319 and +0.0233 (mean +0.0276). These support a cleaning-associated improvement in this setup. Figure 1’s “+0.026” is close but not identical to the CSV mean, likely reflecting a different aggregation/rounding; the bundle does not explain the discrepancy.

In round 2, adding license/provenance filters and 8.6k new rows (U) has per-seed deltas vs C7 of +0.0030 and −0.0029, so the mean change is effectively zero. Removing Kaggle-proxy rows (UK) moves differently across seeds. Keeping only the 70% most train-like examples (US) scores 0.4877 mean vs C7 0.4920 (−0.0043), consistent with the figure’s reported roughly −0.004. Figure 4 additionally reports about −0.001 for removing roughly 70k proxy rows and −0.016 weak-label recall for the train-like subset; those exact figures cannot be independently reconstructed from the supplied round-2 CSV.

### Round 4 paired results and uncertainty

Publisher paired-file estimates and 95% intervals: prompt-first sampling P +0.89 pp [−0.48,+2.22]; typography matching T +2.63 pp [+1.19,+3.93]; Vertex-only Gemini 3.7 G +0.93 pp [0.00,+1.86]; adding the prvsiyan extended rows X −0.92 pp [−2.26,+0.41]. Paired intervals are shown in [`06_dataq4_paired_intervals.svg`](analysis-output/openpool_v11/figures/06_dataq4_paired_intervals.svg). The paired file also reports prompt-centered/half-B metrics; several intervals cross zero. The exact definitions of those metric names are not fully specified in the CSV.

Important reconciliation issue: direct seed differences computed from `dataq4_runs.csv` give mean deltas P +0.685 pp, T +1.760 pp, G +0.630 pp, and X −0.295 pp. The paired-file point estimates differ (especially T and X). For T, the two run deltas are +1.33 and +2.19 pp, neither consistent with a paired estimate of +2.63 pp as a simple average; G’s +0.93 differs from +0.63, and X’s −0.92 differs from −0.295. The report preserves both sources rather than silently choosing one. The files do not provide enough detail to establish whether paired estimates use a different resampling unit, estimator, baseline, or experiment version. See [`dataq4_reconciliation.csv`](analysis-output/openpool_v11/tables/dataq4_reconciliation.csv).

There are only two seeds. `dataq4_paired.csv` supplies intervals, but the artifacts do not clearly document their construction enough to reconstruct them. Overlap of a reported interval with zero should therefore be treated cautiously, especially for G at the zero boundary. Figure 2’s Spearman 0.64 between label cleaning gains and junk/off-genre share cannot be recalculated because the per-label removed-junk counts are not supplied. Figure 5’s AUC comparison and “2,699 official prompts unseen during fit” are author-reported claims; the local data do not include the experiment implementation/splits or raw predictions needed for independent verification.

## 9. Statistical biases and risks

1. **Source shortcut:** source/model Cramér’s V=0.336; source provenance can predict the label. Random row splits can put the same source artifacts on both sides.
2. **Prompt leakage and imbalance:** 31,355 canonical prompt values but only 13,021 recovered texts; prompt panel completeness is highly variable. Random passage splits may share the same prompt across train/evaluation. `prompt_group` and `prompt_canon` are related identifiers, not proof of independent prompt content.
3. **Class imbalance:** 6.57× row-count range; weak labels may be overwhelmed unless sampling/metrics are class-balanced.
4. **Label-route uncertainty:** Gemini 3.7 has disputed API routing and no `origin_hint` for retained rows. Treating it as unquestioned ground truth could cap achievable accuracy or misstate model-specific style.
5. **Correlated mirrors:** `n_sources` measures multiple source appearances; counting every row as independent can overstate effective evidence. Match weights reduce overall ESS to about 55% of nominal rows.
6. **Experiment selection:** many variants are compared against a 2,699-row official train set, with only two seeds. Reusing it to choose filters/weights can make reported best results optimistic.
7. **Multiple testing and dependence:** the style/length association tests have tiny nominal p-values, but passage rows cluster by source and prompt, and many features were examined. Effect sizes and grouped resampling matter more than row-level p-values.
8. **Matching scope:** `match_weight` brings measured marker margins close to official-train margins but may change source/class composition and does not establish joint covariate balance.
9. **No dates/version field:** temporal drift and exact model snapshot cannot be analyzed from the main table.

Complete prompt panels are important when the question is comparative model style under identical prompts. They are not strictly required for every classification experiment, but prompt-grouped splits and a complete-panel metric are important controls. Stronger future evaluations should hold out both prompt groups and source datasets (or at minimum report each separately), publish split IDs, and use multiple seeds.

## 10. Conclusions and recommendations

### Explicit answers

1. **How reliable are the 190k labels?** Their per-row correctness is not measurable from this bundle. The files support coherent accounting and provenance for much of the corpus, but labels are not independently authenticated. Confidence is lower for disputed Gemini 3.7 routing and rows without traceable provenance. Treat labels as weak/noisy labels and preserve uncertainty.
2. **Which models have strongest and weakest data coverage?** By raw passages, Gemini 3.8 Flash is largest (21,550) and Gemini 3.7 Flash smallest (3,278); Qwen3.8-max is next-smallest (4,271). By recovered prompt coverage, Gemini 3.8 is strongest (66.3%) and Gemini 3.7 weakest (23.9%), followed by Qwen3.8-max (26.7%).
3. **Which sources are most/least trustworthy?** No source can be certified from these files. Explicitly named/route-described sources with provenance are more auditable; the Gemini 3.7 group, empty `origin_hint`, sidecar rows, and unverified one-label datasets warrant more scrutiny. Single-label purity alone is not trust evidence.
4. **How similar is Open Pool to official Kaggle distribution?** It shares all 18 labels and short-text task format, but is far larger, strongly class-imbalanced, source-mixed, and measurably different in style. Exact text overlap is zero; no direct prompt join is available. Independent lexical similarity to official passages is low under the measured method, but that does not prove prompt disjointness.
5. **What explains typography matching’s reported improvement?** The official train and Open Pool differ on measured marker rates; weighting narrows those marginal gaps and the publisher reports a +2.63 pp paired gain. This is a plausible mechanism, but the run table’s two-seed mean is +1.76 pp and does not reconcile with the paired point estimate. Source/prompt composition could explain part of either result.
6. **How important is complete prompt coverage?** It enables controlled same-prompt comparisons and reduces prompt-content confounding. Only 592/13,021 recovered prompts are complete, so it is valuable as a controlled slice but not representative enough to be the sole evaluation set. Prompt-grouped splits are more broadly important than requiring every prompt to have all 18 labels.
7. **What biases can mislead validation?** Source and prompt leakage, label imbalance/noise, multiple mirrors, repeated tuning on the small official set, only two seeds, and unadjusted row-level significance tests.
8. **Is dataset size more valuable than quality?** The experiments suggest cleaning can matter: at equal 45k rows, cleaned data improved over raw by about 2.7 pp. Adding more filtered rows did not consistently improve the score; removing a 30% subset reduced it modestly. This suggests quality/provenance and representativeness can outweigh raw quantity at the tested size, but it is not a universal size-vs-quality curve.
9. **Which findings are convincing versus uncertain?** Strong descriptive facts include row/source/class counts, exact prompt reconciliation, panel-gap scope, duplicate checks, and source-label association. Round-1 cleaning gain is consistent across two seeds but still limited. Round-4 effects and intervals are less certain because of only two seeds and unreconciled paired/run estimates. Individual label correctness, Gemini routing/AUC claims, exact prompt overlap, and causal attribution of typography gains remain unverified.
10. **Most promising future uses?** Use the pool for class-balanced augmentation, source-held-out and prompt-held-out robustness experiments, weak-label pretraining/fine-tuning with uncertain classes flagged, and style-matching ablations. Keep provenance fields, test with/without Gemini 3.7, compare equal-sized clean vs noisy subsets, publish split IDs and raw predictions, and report both complete-panel and grouped external evaluation. Do not let source identity or mirrors cross evaluation boundaries.

### Supporting outputs

- Inventory: [`file_inventory.csv`](analysis-output/openpool_v11/tables/file_inventory.csv)
- Counts and coverage: [`label_distribution_and_coverage.csv`](analysis-output/openpool_v11/tables/label_distribution_and_coverage.csv), [`prompt_label_coverage.csv`](analysis-output/openpool_v11/tables/prompt_label_coverage.csv)
- Sources and provenance: [`source_quality_signals.csv`](analysis-output/openpool_v11/tables/source_quality_signals.csv), [`publisher_sources_reconciliation.csv`](analysis-output/openpool_v11/tables/publisher_sources_reconciliation.csv)
- Experiments: [`author_experiment_summary.csv`](analysis-output/openpool_v11/tables/author_experiment_summary.csv), [`dataq4_reconciliation.csv`](analysis-output/openpool_v11/tables/dataq4_reconciliation.csv)
- Main figures: [label counts](analysis-output/openpool_v11/figures/01_label_counts.svg), [top sources](analysis-output/openpool_v11/figures/02_top_sources.svg), [prompt panels](analysis-output/openpool_v11/figures/03_prompt_panel_counts.svg), [style markers](analysis-output/openpool_v11/figures/04_style_marker_zscores.svg), [weight distribution](analysis-output/openpool_v11/figures/05_match_weight_distribution.svg), [experiment intervals](analysis-output/openpool_v11/figures/06_dataq4_paired_intervals.svg)

