# E31 Inference Benchmark

The benchmark uses the same deterministic 4,000-row sample and sampling seed as E30. The E30 individual checkpoint measurements are reused for the unchanged 17-member baseline. Each of the 23 additional E31 checkpoints is measured locally, then the complete fixed 40-member roster is run sequentially on one RTX 5070 Ti with per-checkpoint batch sizes and OOM reduction enabled.

The report records per-checkpoint effective batch size, peak allocated/reserved VRAM, model-load time, pure inference time, cleanup time and total time. It also measures the E28-compatible E19-F-A/F-C grouping embedding, candidate retrieval/scoring/clustering and Hungarian assignment stages on the same sample. Two-GPU scheduling uses measured checkpoint runtimes and the E28 longest-runtime-first scheduler. The 7.2-hour ceiling includes the measured grouping/assignment overhead; the local RTX-to-T4 extrapolation and hidden Kaggle test-row uncertainty are reported explicitly.

**Benchmark is pending the final E31 roster.** Measurements will be written to `artifacts/analysis/e31/inference_benchmark.json` and summarized below when complete.
