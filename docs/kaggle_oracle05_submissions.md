# ORACLE05 Kaggle submissions

Three ORACLE05 variants were submitted through completed Kaggle code kernels. The competition accepted each `submission.parquet`. All three submissions are complete; Kaggle has not reported public scores.

| Strategy | Cross-fitted V0 balanced accuracy | Full-V0 diagnostic | Kernel version | Submission | Status | Public score |
|---|---:|---:|---|---:|---|---|
| A — fixed label-pair rules | 67.167% | 67.167% | `zodrak/oracle05-a-pair-rules` v2 | 58479298 | Complete | Not reported |
| B — probability-margin rules | 67.686% | 67.723% | `zodrak/oracle05-b-margin` v3 | 58479299 | Complete | Not reported |
| C — checkpoint-informed rules | 67.906% | 69.614% | `zodrak/oracle05-c-checkpoint` v2 | 58479301 | Complete | Not reported |

All three kernels completed and their downloaded outputs passed schema and row-order checks: three rows, exactly `id,model`, unique IDs, and canonical labels. All three output files have the same SHA-256 (`0cc980a510ebb3513dbeb259ca79dd818cd412b119c547ca40a2da13409feaaa`) and predict `glm-5.3`, `gpt-5.5`, and `gemini-3.1-pro-preview` for the three visible test IDs. The ORACLE05 rules did not change any visible test prediction, so these submissions are operationally identical on the available test rows.

The exact kernel versions, submission descriptions, checkpoint roster, probability-cache hashes and submission statuses are recorded in `artifacts/analysis/oracle05/kaggle_submissions/submission_manifest.json`. Kernel output files and logs are under `artifacts/analysis/oracle05/kaggle_submissions/kernel_outputs/`.
