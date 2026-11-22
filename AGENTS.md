# Repository workflow instructions

## Experiment catalogue

- Register every new ensemble or evaluation experiment in `configs/ensemble_catalog_v2.json` as part of the initial implementation, before running training or evaluation.
- Use a planned/running status initially, then update the same catalogue entry with its results, status, and artifact/documentation paths when available.
- The dashboard reads this JSON at runtime; keep the catalogue entry complete enough to appear there without a separate user request.
