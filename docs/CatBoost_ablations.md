# CatBoost stylometry feature ablations

## Status and goal

The initial ablation suite was run on V0 and is retained under `analysis-output/catboost_ablations/` as historical results. The V1 suite is retained under `analysis-output/catboost_ablations_v1/`. **Those historical V1 runs used the old loader behavior and trained on V0 train plus Open Pool rows (50,957 rows); they are mixed-data results, not Open-Pool-only V1 results.** After the pipeline correction, V1 training means the 48,798 Open Pool rows only, with the same 540 V0 validation rows. The old V1 ablation scores must not be presented as results for this corrected V1 definition. The old suite's A0 scored 0.2907 balanced accuracy; A4 selected baseline + expanded function words (F) + Unicode indicators (U), with 119 features and 0.3352 balanced accuracy. The top candidate, A3-minus-U, scored 0.3444; the repeated A5 splits favored A4 over A0. Full historical results are in `analysis-output/catboost_ablations_v1/report.md`. The current CatBoost model is a baseline using 35 numeric features in `src/feature_factory.py::stylometry` (6 length/sentence, 2 case/digit, 11 punctuation, and 16 function-word features); the earlier 27-feature count was incorrect. Feature ideas below are proposals, not results. The goal is to find out whether stable writing-style statistics improve classification, which groups contribute, and whether their errors complement the existing models.

For corrected V1 A0-A4 runs, use only curated Open Pool training rows and the same 540 held-out V0 validation rows; prompt groups must not cross the validation boundary. For A5, create three deterministic prompt-grouped 80/20 splits of only the Open Pool rows; keep V0 rows excluded. Repeatedly selecting features on the fixed validation set can overfit it, so treat these runs as development evidence, not an unbiased final estimate. Do not fit feature transforms, thresholds, or calibration on validation labels.

## Current feature baseline

`stylometry()` returns, in order:

1. **Length and sentence structure (6):** character count, word count, sentence count, characters per word, words per sentence, and mean word length.
2. **Character rates (2):** uppercase-character fraction and digit-character fraction.
3. **Punctuation rates (11):** per-character fractions for comma, semicolon, colon, em dash, en dash, hyphen, apostrophe, double quote, opening parenthesis, question mark, and exclamation mark.
4. **Function-word frequencies (16):** counts divided by word count for `the, a, an, and, or, but, if, of, to, in, for, with, that, this, is, are`.

The function-word values are word-token frequencies, despite being described as rates. Word tokens use `\b\w+\b`; sentence count splits on runs of `.`, `!`, and `?`. These implementation details should be held fixed for the baseline and recorded if changed.

CatBoost receives these numeric features alone. Current settings are in `configs/models/catboost_stylometry.yaml` (800 iterations, depth 7, learning rate 0.05, early stopping and checkpoint selection as configured there). Preserve the same settings for feature-only ablations so that feature changes are interpretable.

## Proposed feature groups

Add groups behind an explicit feature recipe/configuration. Keep each group independently selectable so ablations do not require editing the baseline definition. Fit any corpus-level vocabulary or thresholds on training rows only.

| ID | Group | Candidate features |
|---|---|---|
| B | Existing baseline | The 35 features above. |
| L | Length distributions | Median, standard deviation, and selected quantiles (for example p10/p90) of word length and sentence length; fraction of very short/long sentences; paragraph count and paragraph-length summaries. Use robust summaries and define behavior for empty text. |
| V | Vocabulary | Type-token ratio with a length-aware alternative such as MTLD or moving-average TTR; hapax fraction; repeated-token fraction; mean/variance of token frequency within the passage. Clearly define tokenization and avoid unstable raw vocabulary counts as the only signal. |
| C | Capitalization | Lowercase/uppercase/title-case word fractions; all-caps token fraction; sentence-initial capitalization; internal capitals; capitalization after punctuation. |
| P | Punctuation and typography | Counts and rates for all common punctuation, brackets, curly quotes, ellipses, repeated punctuation, colon/semicolon use, and dash types; punctuation runs and punctuation per sentence. Keep Unicode normalization explicit. |
| W | Whitespace and layout | Newline and blank-line counts/rates, indentation, tabs, leading/trailing spaces, repeated spaces, and paragraph/sentence boundary spacing. |
| F | Expanded function words | Frequencies for a predeclared, linguistically motivated list of closed-class words, contractions, and discourse markers. Freeze the list before evaluation; avoid selecting words by validation performance. |
| R | Readability and syntax proxies | Sentence-length dispersion, word-length dispersion, clause/connector proxies, passive-voice proxy only if deterministic and dependency-free, and established readability formulas where their assumptions fit the text. Treat these as noisy proxies, not true syntax parses. |
| X | Interactions | A small predeclared set of normalized combinations such as punctuation-per-sentence and long-word fraction. CatBoost can learn interactions, so only retain engineered interactions if they improve validation evidence or stability. |
| M | Markdown and list formatting | Bullet and numbered-list rates, Markdown heading rate, and fenced-code count. Evaluate separately because these may reflect source formatting. |
| U | Unicode formatting indicators | Non-ASCII and NFKC-change rates. Evaluate separately because they may reflect source or language distribution. |

Potentially confounded features should be evaluated separately: paragraph/list formatting, markdown/code markers, non-ASCII rates, and other source-format indicators may identify collection or prompt artifacts rather than author style. Do not add prompt IDs, labels encoded from source metadata, row IDs, or features computed from the full corpus. Never use test-set statistics to fit transformations.

## Ablation sequence

Use identical row IDs, labels, V1 split, CatBoost parameters, class weighting, random seed, stopping rule, and calibration procedure in each A0-A4 comparison. Save validation probabilities aligned by row ID, feature names/version, config, and code revision. First run all features with no feature selection; then do group ablations.

| Run | Feature recipe | Question |
|---|---|---|
| A0 | B | Reproduce and record the current baseline. |
| A1 | B + each of L, V, C, P, W, F, R, X, M, U individually | Which single group adds useful signal, and do formatting confounds behave differently? |
| A2 | B minus each existing subgroup (length/sentence, case/digits, punctuation, function words) | Does any current group hurt or add no value? |
| A3 | B + promising groups, one group removed at a time | Are selected groups redundant or complementary? |
| A4 | Best compact recipe from A1–A3 | Does a smaller recipe match or beat the full candidate set? |
| A5 | A0 and A4 across several fixed group splits/seeds | Are conclusions stable across prompt groups? |
| A6 | A0/A4 paired with SVC and existing neural validation predictions | Does CatBoost add complementary errors or improve an existing blend? |

Do not search many CatBoost hyperparameters in the feature-ablation stage. If A0 cannot be reproduced, first resolve the discrepancy. After identifying promising recipes, run a small, predeclared parameter check (for example depth 5/7 and regularization around the current value) as a separate experiment, without mixing it into feature attribution.

## Metrics and reporting

Report at least:

- **Primary:** validation balanced accuracy, matching the project’s model-selection convention.
- **Supporting:** ordinary accuracy, macro F1, log loss if calibrated probabilities are available, and per-class recall.
- Confusion matrix and per-class changes relative to A0.
- Feature count and CatBoost training/inference time.
- Paired row-level correctness changes and prompt-group-level score variation for multi-split runs.
- For A6, equal-weight blend score and the incremental change when adding CatBoost to the same existing model set. Do not fit blend weights on validation labels.

Use a paired prompt-group bootstrap or repeated fixed grouped splits to show uncertainty once a short list of recipes is selected. Resample whole prompt groups, not individual rows. Report the number of groups and intervals; do not present a tiny score difference on the single 540-row validation set as decisive. Keep a final untouched evaluation set or competition leaderboard for confirmation if one is available, and do not repeatedly tune against leaderboard feedback.

## Recommended implementation order

1. Add a feature recipe argument to the stylometry extractor while preserving the exact existing baseline as the default. This implementation is in place; the historical baseline checkpoint reproduces exactly when using the default recipe.
2. Record the V1 A0 run with saved predictions and config. Confirm it matches the existing V1 CatBoost baseline before running other recipes.
3. Implement L, V, C, P, and W as independent groups first; these are deterministic and inexpensive. Add F and R only after their definitions are fixed and reviewed.
4. Run A1 and A2, then A3/A4. Inspect class and group-level errors before expanding the search.
5. Run A5/A6 only for the small set of promising recipes. Promote a feature recipe only when the gain is repeatable, useful to the ensemble or a clear standalone improvement, and not driven by obvious formatting/source leakage.

The pipeline retains the model ID `catboost_stylometry`; the dashboard and registry distinguish ablation runs by their experiment config and description. Candidate groups are now implemented in the feature extractor, but are not considered useful until the documented runs are complete and reported.
