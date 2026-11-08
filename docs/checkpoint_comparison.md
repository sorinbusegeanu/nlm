# Checkpoint comparison report

This report will be generated after the V1-all UltraBERT run finishes. Run this single command from the repository root; it reuses the saved validation predictions and writes the comparison below this placeholder.

```bash
/home/zodrak/cuda13.x/bin/python -m src.pipeline models compare-checkpoints \
  --previous ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0 \
  --new ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2
```

The command replaces this file with the four-section report, including the frozen V0 split hash and both checkpoint IDs.
