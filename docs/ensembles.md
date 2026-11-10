# Ensemble registry

<!-- ensemble-registry:start -->
```yaml
active_candidate_roster_hash: 8294d46b142142d173c4c91083ff03d98311eb28ea34c54dc1a4680ffce98564
active_candidate_roster: &id037
- model: catboost_stylometry
  candidate_id: catboost_two_stage_20261108T0850Z_C_stage2
  training_dataset: v0
  source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_two_stage_20261108T0850Z_C_stage2_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_two_stage/20261108T0850Z/C/stage2/checkpoint
  checkpoint_filename: checkpoint
  selected_epoch: null
  validation_balanced_accuracy: 0.34444444444444444
- model: embeddinggemma2
  candidate_id: embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831
  training_dataset: v1_all
  source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831/checkpoints/embeddinggemma2_foldvalidation
  checkpoint_filename: embeddinggemma2_foldvalidation
  selected_epoch: 9
  validation_balanced_accuracy: 0.4407407407407407
- model: gemma4_e2b
  candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
  training_dataset: v1.1
  source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
  checkpoint_filename: gemma4_e2b_foldvalidation
  selected_epoch: 7
  validation_balanced_accuracy: 0.5277777777777778
- model: granite4_350m
  candidate_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
  training_dataset: v1_all
  source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831/checkpoints/granite4_350m_foldvalidation
  checkpoint_filename: granite4_350m_foldvalidation
  selected_epoch: 5
  validation_balanced_accuracy: 0.48518518518518516
- model: granite4_h_micro
  candidate_id: granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831
  training_dataset: v1_all
  source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831/checkpoints/granite4_h_micro_foldvalidation
  checkpoint_filename: granite4_h_micro_foldvalidation
  selected_epoch: 3
  validation_balanced_accuracy: 0.5277777777777778
- model: llama32_3b
  candidate_id: llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81
  training_dataset: v1_all
  source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81/checkpoints/llama32_3b_foldvalidation
  checkpoint_filename: llama32_3b_foldvalidation
  selected_epoch: 2
  validation_balanced_accuracy: 0.5425925925925926
- model: ministral3_3b
  candidate_id: ministral3_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg4619b410_code279cc81
  training_dataset: v1_all
  source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg4619b410_code279cc81_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg4619b410_code279cc81/checkpoints/ministral3_3b_foldvalidation
  checkpoint_filename: ministral3_3b_foldvalidation
  selected_epoch: 3
  validation_balanced_accuracy: 0.5203703703703703
- model: qwen06
  candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
  training_dataset: v1.5
  source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
  checkpoint_filename: qwen06_foldvalidation
  selected_epoch: 6
  validation_balanced_accuracy: 0.47777777777777775
- model: qwen17
  candidate_id: qwen17_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgd07c657b_code279cc81
  training_dataset: v1_all
  source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgd07c657b_code279cc81_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgd07c657b_code279cc81/checkpoints/qwen17_foldvalidation
  checkpoint_filename: qwen17_foldvalidation
  selected_epoch: 3
  validation_balanced_accuracy: 0.524074074074074
- model: qwen4
  candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
  training_dataset: v0
  source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
  checkpoint_filename: qwen4_foldvalidation
  selected_epoch: 9
  validation_balanced_accuracy: 0.2833333333333333
- model: smollm3_3b
  candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
  training_dataset: v1.5
  source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
  checkpoint_filename: smollm3_3b_foldvalidation
  selected_epoch: 5
  validation_balanced_accuracy: 0.5074074074074074
- model: svc
  candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
  training_dataset: v1
  source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
  checkpoint_filename: svc_foldvalidation.joblib
  selected_epoch: null
  validation_balanced_accuracy: 0.4166666666666667
- model: ultrabert380
  candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
  training_dataset: v0
  source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_manifest.json
  checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff/checkpoints/ultrabert380_continued
  checkpoint_filename: ultrabert380_continued
  selected_epoch: 2
  validation_balanced_accuracy: 0.5518518518518518
checkpoint_replacement_policy: replace a model checkpoint only when validation balanced
  accuracy is higher
reliability_fit_split: V0 train
last_recorded_active_roster_hash: 8294d46b142142d173c4c91083ff03d98311eb28ea34c54dc1a4680ffce98564
optimizer_protocol: forward_group_cv_weighted_v1
ensembles:
- version: v1
  created_at_utc: '2026-11-03'
  candidate_roster_hash: 49464fbba9b7d20b31e20ce1bb788c773c9665f388c14a92ecfcc476c92e8442
  candidate_pool_count: 18
  candidate_checkpoints:
  - candidate_id: catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfgfa54eec9_code667e2d0
    model: catboost_stylometry
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfgfa54eec9_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3055555555555555
  - candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    model: catboost_stylometry
    training_dataset: v1
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - candidate_id: embeddinggemma2_v0_continue10_fromeab947d3_seed2026_cfg2ba833a7_code667e2d0
    model: embeddinggemma2
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v0_continue10_fromeab947d3_seed2026_cfg2ba833a7_code667e2d0/checkpoints/embeddinggemma2_continued
    checkpoint_filename: embeddinggemma2_continued
    selected_epoch: null
    validation_balanced_accuracy: 0.22777777777777775
  - candidate_id: embeddinggemma2_v1_datada64f007_split147ce64e_seed2026_cfgdfb26ed6_code667e2d0
    model: embeddinggemma2
    training_dataset: v1
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1_datada64f007_split147ce64e_seed2026_cfgdfb26ed6_code667e2d0/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.4092592592592592
  - candidate_id: embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0
    model: embeddinggemma2
    training_dataset: v1.5
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.41851851851851857
  - candidate_id: granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0
    model: granite4_h_micro
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.31481481481481477
  - candidate_id: llama32_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg3d4f188e_code667e2d0
    model: llama32_3b
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg3d4f188e_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.34814814814814815
  - candidate_id: ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0
    model: ministral3_3b
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.37407407407407406
  - candidate_id: qwen06_best_epoch03_v0_from3789_code8422c8c
    model: qwen06
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v0_data40a22ce8_split147ce64e_seed2026_cfg42943789_code8422c8c/checkpoints/qwen06_foldvalidation/epoch-03
    checkpoint_filename: epoch-03
    selected_epoch: null
    validation_balanced_accuracy: 0.36111111111111116
  - candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    model: qwen06
    training_dataset: v1.5
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.47777777777777775
  - candidate_id: qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfgf7e4738d_code667e2d0
    model: qwen17
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfgf7e4738d_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.31666666666666665
  - candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    model: qwen4
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.2833333333333333
  - candidate_id: smollm3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg58d8423d_code667e2d0
    model: smollm3_3b
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg58d8423d_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.30925925925925923
  - candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    model: smollm3_3b
    training_dataset: v1.5
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.5074074074074074
  - candidate_id: svc_v0_data40a22ce8_split147ce64e_seed2026_cfg6c3be430_code365dbc7
    model: svc
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v0_data40a22ce8_split147ce64e_seed2026_cfg6c3be430_code365dbc7/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.3074074074074074
  - candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    model: svc
    training_dataset: v1
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - candidate_id: ultrabert380_v0_data40a22ce8_split147ce64e_seed2026_cfg5a387d29_code365dbc7
    model: ultrabert380
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v0_data40a22ce8_split147ce64e_seed2026_cfg5a387d29_code365dbc7/checkpoints/ultrabert380_foldvalidation
    checkpoint_filename: ultrabert380_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.34444444444444444
  - candidate_id: ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0
    model: ultrabert380
    training_dataset: v1
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0/checkpoints/ultrabert380_foldvalidation
    checkpoint_filename: ultrabert380_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.5111111111111111
  selected_strategy: equal_probability_blend
  selected_members:
  - candidate_id: catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfgfa54eec9_code667e2d0
    model: catboost_stylometry
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfgfa54eec9_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3055555555555555
  - candidate_id: ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0
    model: ministral3_3b
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.37407407407407406
  - candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    model: qwen06
    training_dataset: v1.5
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.47777777777777775
  - candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    model: smollm3_3b
    training_dataset: v1.5
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.5074074074074074
  - candidate_id: svc_v0_data40a22ce8_split147ce64e_seed2026_cfg6c3be430_code365dbc7
    model: svc
    training_dataset: v0
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v0_data40a22ce8_split147ce64e_seed2026_cfg6c3be430_code365dbc7/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.3074074074074074
  - candidate_id: ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0
    model: ultrabert380
    training_dataset: v1
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0/checkpoints/ultrabert380_foldvalidation
    checkpoint_filename: ultrabert380_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.5111111111111111
  predicted_balanced_accuracy: 0.6
  validation_accuracy: 0.6
  submitted_balanced_accuracy: 0.546
  prediction_experiment_id: prediction_v0_49eb0554416f
  prediction_manifest: /home/zodrak/nlm/artifacts/predictions/prediction_v0_49eb0554416f_manifest.json
  validation_rows: 540
  historical_note: Original v1 evaluated 18 model/version candidates; its candidate
    pool is retained here as a historical record.
- version: v2
  created_at_utc: '2026-11-05T09:54:34.267379+00:00'
  trigger_experiment_id: null
  candidate_roster_hash: 68eb2c9deb80230ebd5e3d290e02d1ddfc03c9426d8684044bdbdafd6f14c678
  candidate_checkpoints:
  - &id001
    model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - &id002
    model: embeddinggemma2
    candidate_id: embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 8
    validation_balanced_accuracy: 0.41851851851851857
  - &id003
    model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.31481481481481477
  - model: llama32_3b
    candidate_id: llama32_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg3d4f188e_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg3d4f188e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg3d4f188e_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 10
    validation_balanced_accuracy: 0.34814814814814815
  - &id004
    model: ministral3_3b
    candidate_id: ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.37407407407407406
  - &id005
    model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - &id006
    model: qwen17
    candidate_id: qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfgf7e4738d_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfgf7e4738d_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfgf7e4738d_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.31666666666666665
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - &id007
    model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - &id008
    model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - &id009
    model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  selected_strategy: equal_probability_blend
  selected_members:
  - *id001
  - *id002
  - *id003
  - *id004
  - *id005
  - *id006
  - *id007
  - *id008
  - *id009
  predicted_balanced_accuracy: 0.5981481481481482
  validation_accuracy: 0.5981481481481481
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_2b121104edd1
  prediction_manifest: artifacts/predictions/prediction_v0_2b121104edd1_manifest.json
  validation_rows: 540
- version: v3
  created_at_utc: '2026-11-05T10:49:33.134326+00:00'
  trigger_experiment_id: granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0
  candidate_roster_hash: c86f9005d8ad5982e905be38aa35c59928aeb5f2616de3a4234912afcfe9866f
  candidate_checkpoints:
  - &id010
    model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - &id011
    model: embeddinggemma2
    candidate_id: embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 8
    validation_balanced_accuracy: 0.41851851851851857
  - &id012
    model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_350m
    candidate_id: granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 4
    validation_balanced_accuracy: 0.03148148148148148
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.31481481481481477
  - model: llama32_3b
    candidate_id: llama32_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg3d4f188e_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg3d4f188e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg3d4f188e_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 10
    validation_balanced_accuracy: 0.34814814814814815
  - &id013
    model: ministral3_3b
    candidate_id: ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.37407407407407406
  - &id014
    model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - &id015
    model: qwen17
    candidate_id: qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfgf7e4738d_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfgf7e4738d_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfgf7e4738d_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.31666666666666665
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - &id016
    model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - &id017
    model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - &id018
    model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  selected_strategy: equal_probability_blend
  selected_members:
  - *id010
  - *id011
  - *id012
  - *id013
  - *id014
  - *id015
  - *id016
  - *id017
  - *id018
  predicted_balanced_accuracy: 0.5981481481481482
  validation_accuracy: 0.5981481481481481
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_7976fb196c8d
  prediction_manifest: artifacts/predictions/prediction_v0_7976fb196c8d_manifest.json
  validation_rows: 540
- version: v4
  created_at_utc: '2026-11-07T18:58:56.947114+00:00'
  trigger_experiment_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
  candidate_roster_hash: b2a290304055198e65fe02af046c81c1c22edb2f95f15f048c8a5f1525005437
  candidate_checkpoints:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: embeddinggemma2
    candidate_id: embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 8
    validation_balanced_accuracy: 0.41851851851851857
  - &id019
    model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - &id020
    model: granite4_350m
    candidate_id: granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 4
    validation_balanced_accuracy: 0.4388888888888889
  - &id021
    model: granite4_h_micro
    candidate_id: granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.31481481481481477
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - &id022
    model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - &id023
    model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: qwen17
    candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.4629629629629629
  - &id024
    model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - &id025
    model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - &id026
    model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - &id027
    model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  selected_strategy: equal_probability_blend
  selected_members:
  - *id019
  - *id020
  - *id021
  - *id022
  - *id023
  - *id024
  - *id025
  - *id026
  - *id027
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_87730c618985
  prediction_manifest: /home/zodrak/nlm/artifacts/predictions/prediction_v0_87730c618985_manifest.json
  validation_rows: 540
- version: v5
  created_at_utc: '2026-11-08T02:54:49.391686+00:00'
  trigger_experiment_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
  candidate_roster_hash: f2752084813d54f83b54918f5b1abf965d245087c792999dee6bce59527eb66a
  candidate_checkpoints:
  - &id028
    model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: embeddinggemma2
    candidate_id: embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 8
    validation_balanced_accuracy: 0.41851851851851857
  - &id029
    model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_350m
    candidate_id: granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 4
    validation_balanced_accuracy: 0.4388888888888889
  - &id030
    model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - &id031
    model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - &id032
    model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - &id033
    model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: qwen17
    candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.4629629629629629
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - &id034
    model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - &id035
    model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - &id036
    model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  selected_strategy: equal_probability_blend
  selected_members:
  - *id028
  - *id029
  - *id030
  - *id031
  - *id032
  - *id033
  - *id034
  - *id035
  - *id036
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_c770dff0b016
  prediction_manifest: artifacts/predictions/prediction_v0_c770dff0b016_manifest.json
  validation_rows: 540
- version: v6
  created_at_utc: '2026-11-08T05:23:04.151109+00:00'
  trigger_experiment_id: null
  candidate_roster_hash: f2752084813d54f83b54918f5b1abf965d245087c792999dee6bce59527eb66a
  candidate_checkpoints:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: embeddinggemma2
    candidate_id: embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 8
    validation_balanced_accuracy: 0.41851851851851857
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_350m
    candidate_id: granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 4
    validation_balanced_accuracy: 0.4388888888888889
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: qwen17
    candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.4629629629629629
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  selected_strategy: equal_probability_blend
  selected_members:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  group_oof_metrics:
    previous_champion:
      accuracy: 0.6
      balanced_accuracy: 0.6000000000000001
      log_loss: 1.1764712402418676
      ensemble_size: 9
    equal_probability_blend:
      accuracy: 0.562962962962963
      balanced_accuracy: 0.5629629629629629
      log_loss: 1.2439472447437963
      ensemble_size: 10
    optimized_probability_blend:
      accuracy: 0.5370370370370371
      balanced_accuracy: 0.537037037037037
      log_loss: 1.2010960067975542
      ensemble_size: 10
  selected_weights:
    catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0: 0.1111111111111111
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.1111111111111111
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.1111111111111111
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.1111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.1111111111111111
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.1111111111111111
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.1111111111111111
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.1111111111111111
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.1111111111111111
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_27e1439c9742
  prediction_manifest: artifacts/predictions/prediction_v0_27e1439c9742_manifest.json
  validation_rows: 540
  selected_comparison: previous_champion
  optimizer_protocol: forward_group_cv_weighted_v1
  individual_accuracy:
    catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0: 0.34444444444444444
    embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0: 0.4185185185185185
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.5277777777777778
    granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0: 0.4388888888888889
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.5148148148148148
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.5111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.5092592592592593
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.4777777777777778
    qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0: 0.46296296296296297
    qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0: 0.2833333333333333
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.5074074074074074
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.42962962962962964
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.5314814814814814
  forward_selection_steps:
  - step: 1
    added_candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    accuracy: 0.5314814814814814
    log_loss: 13.499685231497619
  - step: 2
    added_candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    accuracy: 0.5388888888888889
    log_loss: 2.145115570629925
  - step: 3
    added_candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    accuracy: 0.562962962962963
    log_loss: 1.3218938920400825
  - step: 4
    added_candidate_id: embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0
    accuracy: 0.5666666666666667
    log_loss: 1.2508851432339219
  - step: 5
    added_candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    accuracy: 0.5703703703703704
    log_loss: 1.1964635679066657
  - step: 6
    added_candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    accuracy: 0.5777777777777777
    log_loss: 1.1750316221267798
  - step: 7
    added_candidate_id: granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0
    accuracy: 0.5777777777777777
    log_loss: 1.1696117942021995
  - step: 8
    added_candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    accuracy: 0.5796296296296296
    log_loss: 1.1899919480111998
  - step: 9
    added_candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    accuracy: 0.5796296296296296
    log_loss: 1.1743364385215964
  - step: 10
    added_candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    accuracy: 0.5814814814814815
    log_loss: 1.202693369493464
  optimization_summary: artifacts/predictions/prediction_v0_27e1439c9742_summary.json
  forward_steps_artifact: artifacts/predictions/prediction_v0_27e1439c9742_forward_steps.csv
- version: v7
  created_at_utc: '2026-11-08T11:22:30.802955+00:00'
  trigger_experiment_id: ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2
  candidate_roster_hash: 019051a168b31d7fa9157b1cf9f0f699436cc4db3e34a392817c63e66daa93a5
  candidate_checkpoints:
  - model: catboost_stylometry
    candidate_id: catboost_two_stage_20261108T0850Z_C_stage2
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_two_stage_20261108T0850Z_C_stage2_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_two_stage/20261108T0850Z/C/stage2/checkpoint
    checkpoint_filename: checkpoint
    selected_epoch: null
    validation_balanced_accuracy: 0.34444444444444444
  - model: embeddinggemma2
    candidate_id: embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 8
    validation_balanced_accuracy: 0.41851851851851857
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_350m
    candidate_id: granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 4
    validation_balanced_accuracy: 0.4388888888888889
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: qwen17
    candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.4629629629629629
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  selected_strategy: equal_probability_blend
  selected_comparison: previous_champion
  selected_members:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  validation_metric_scope: prompt_group_out_of_fold
  full_data_metrics_exploratory:
    accuracy: 0.6
    balanced_accuracy: 0.6000000000000001
    log_loss: 1.1764712402418678
  group_oof_metrics:
    equal_probability_blend:
      accuracy: 0.5648148148148148
      balanced_accuracy: 0.5648148148148148
      log_loss: 1.2201449783702316
      ensemble_size: 10
    optimized_probability_blend:
      accuracy: 0.5333333333333333
      balanced_accuracy: 0.5333333333333333
      log_loss: 1.1977959611252138
      ensemble_size: 10
    all_candidate_weighted_blend:
      accuracy: 0.5814814814814815
      balanced_accuracy: 0.5814814814814815
      log_loss: 1.1453569054830774
      ensemble_size: 13
    previous_champion:
      accuracy: 0.6
      balanced_accuracy: 0.6000000000000001
      log_loss: 1.1764712402418678
      ensemble_size: 9
  selected_weights:
    catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0: 0.1111111111111111
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.1111111111111111
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.1111111111111111
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.1111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.1111111111111111
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.1111111111111111
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.1111111111111111
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.1111111111111111
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.1111111111111111
  individual_accuracy:
    catboost_two_stage_20261108T0850Z_C_stage2: 0.34444444444444444
    embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0: 0.4185185185185185
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.5277777777777778
    granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0: 0.4388888888888889
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.5148148148148148
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.5111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.5092592592592593
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.4777777777777778
    qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0: 0.46296296296296297
    qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0: 0.2833333333333333
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.5074074074074074
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.42962962962962964
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.5314814814814814
  forward_selection_steps:
  - step: 1
    added_candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    accuracy: 0.5314814814814814
    log_loss: 13.499685231497619
  - step: 2
    added_candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    accuracy: 0.5388888888888889
    log_loss: 2.145115570629925
  - step: 3
    added_candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    accuracy: 0.562962962962963
    log_loss: 1.3218938920400825
  - step: 4
    added_candidate_id: embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0
    accuracy: 0.5666666666666667
    log_loss: 1.2508851432339219
  - step: 5
    added_candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    accuracy: 0.5703703703703704
    log_loss: 1.1964635679066657
  - step: 6
    added_candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    accuracy: 0.5777777777777777
    log_loss: 1.1750316221267798
  - step: 7
    added_candidate_id: granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0
    accuracy: 0.5777777777777777
    log_loss: 1.1696117942021995
  - step: 8
    added_candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    accuracy: 0.5796296296296296
    log_loss: 1.1899919480111998
  - step: 9
    added_candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    accuracy: 0.5796296296296296
    log_loss: 1.1743364385215964
  - step: 10
    added_candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    accuracy: 0.5796296296296296
    log_loss: 1.161659913991977
  optimization_summary: artifacts/predictions/prediction_v0_1fcbfd77614f_summary.json
  forward_steps_artifact: artifacts/predictions/prediction_v0_1fcbfd77614f_forward_steps.csv
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_1fcbfd77614f
  prediction_manifest: artifacts/predictions/prediction_v0_1fcbfd77614f_manifest.json
  validation_rows: 540
  optimizer_protocol: forward_group_cv_weighted_v1
- version: v8
  created_at_utc: '2026-11-08T11:58:42.471807+00:00'
  trigger_experiment_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
  candidate_roster_hash: 2ae458673baede1263c053f9b872678c6a5e432c510c0a7e0745ee761ea8bd26
  candidate_checkpoints:
  - model: catboost_stylometry
    candidate_id: catboost_two_stage_20261108T0850Z_C_stage2
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_two_stage_20261108T0850Z_C_stage2_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_two_stage/20261108T0850Z/C/stage2/checkpoint
    checkpoint_filename: checkpoint
    selected_epoch: null
    validation_balanced_accuracy: 0.34444444444444444
  - model: embeddinggemma2
    candidate_id: embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 8
    validation_balanced_accuracy: 0.41851851851851857
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_350m
    candidate_id: granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 4
    validation_balanced_accuracy: 0.4388888888888889
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: qwen17
    candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.4629629629629629
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 2
    validation_balanced_accuracy: 0.5518518518518518
  selected_strategy: equal_probability_blend
  selected_comparison: previous_champion
  selected_members:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  validation_metric_scope: prompt_group_out_of_fold
  full_data_metrics_exploratory:
    accuracy: 0.6
    balanced_accuracy: 0.6000000000000001
    log_loss: 1.1764712402418678
  group_oof_metrics:
    equal_probability_blend:
      accuracy: 0.5925925925925926
      balanced_accuracy: 0.5925925925925926
      log_loss: 1.2149326575137351
      ensemble_size: 3
    optimized_probability_blend:
      accuracy: 0.5962962962962963
      balanced_accuracy: 0.5962962962962963
      log_loss: 1.2094002192426656
      ensemble_size: 3
    all_candidate_weighted_blend:
      accuracy: 0.5925925925925926
      balanced_accuracy: 0.5925925925925926
      log_loss: 1.1329877794606844
      ensemble_size: 13
    previous_champion:
      accuracy: 0.6
      balanced_accuracy: 0.6000000000000001
      log_loss: 1.1764712402418678
      ensemble_size: 9
  selected_weights:
    catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0: 0.1111111111111111
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.1111111111111111
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.1111111111111111
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.1111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.1111111111111111
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.1111111111111111
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.1111111111111111
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.1111111111111111
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.1111111111111111
  individual_accuracy:
    catboost_two_stage_20261108T0850Z_C_stage2: 0.34444444444444444
    embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0: 0.4185185185185185
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.5277777777777778
    granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0: 0.4388888888888889
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.5148148148148148
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.5111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.5092592592592593
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.4777777777777778
    qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0: 0.46296296296296297
    qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0: 0.2833333333333333
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.5074074074074074
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.42962962962962964
    ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff: 0.5518518518518518
  forward_selection_steps:
  - step: 1
    added_candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    accuracy: 0.5518518518518518
    log_loss: 2.5716356138271537
  - step: 2
    added_candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    accuracy: 0.5796296296296296
    log_loss: 1.323572680003492
  - step: 3
    added_candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    accuracy: 0.6018518518518519
    log_loss: 1.250228507537642
  optimization_summary: artifacts/predictions/prediction_v0_40beb809dc91_summary.json
  forward_steps_artifact: artifacts/predictions/prediction_v0_40beb809dc91_forward_steps.csv
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_40beb809dc91
  prediction_manifest: artifacts/predictions/prediction_v0_40beb809dc91_manifest.json
  validation_rows: 540
  optimizer_protocol: forward_group_cv_weighted_v1
- version: v9
  created_at_utc: '2026-11-08T21:32:17.460260+00:00'
  trigger_experiment_id: embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831
  candidate_roster_hash: d7f775082e4ad123e5bdb620174b2b4607eb8771f1e7f4533e900cc460aac69e
  candidate_checkpoints:
  - model: catboost_stylometry
    candidate_id: catboost_two_stage_20261108T0850Z_C_stage2
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_two_stage_20261108T0850Z_C_stage2_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_two_stage/20261108T0850Z/C/stage2/checkpoint
    checkpoint_filename: checkpoint
    selected_epoch: null
    validation_balanced_accuracy: 0.34444444444444444
  - model: embeddinggemma2
    candidate_id: embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.4407407407407407
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_350m
    candidate_id: granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 4
    validation_balanced_accuracy: 0.4388888888888889
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: qwen17
    candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.4629629629629629
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 2
    validation_balanced_accuracy: 0.5518518518518518
  selected_strategy: equal_probability_blend
  selected_comparison: previous_champion
  selected_members:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  validation_metric_scope: prompt_group_out_of_fold
  full_data_metrics_exploratory:
    accuracy: 0.6
    balanced_accuracy: 0.6000000000000001
    log_loss: 1.1764712402418678
  group_oof_metrics:
    equal_probability_blend:
      accuracy: 0.5888888888888889
      balanced_accuracy: 0.5888888888888889
      log_loss: 1.2216654501671351
      ensemble_size: 3
    optimized_probability_blend:
      accuracy: 0.5907407407407408
      balanced_accuracy: 0.5907407407407407
      log_loss: 1.218002871794561
      ensemble_size: 3
    all_candidate_weighted_blend:
      accuracy: 0.5907407407407408
      balanced_accuracy: 0.5907407407407407
      log_loss: 1.129450232872043
      ensemble_size: 13
    previous_champion:
      accuracy: 0.6
      balanced_accuracy: 0.6000000000000001
      log_loss: 1.1764712402418678
      ensemble_size: 9
  selected_weights:
    catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0: 0.1111111111111111
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.1111111111111111
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.1111111111111111
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.1111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.1111111111111111
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.1111111111111111
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.1111111111111111
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.1111111111111111
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.1111111111111111
  individual_accuracy:
    catboost_two_stage_20261108T0850Z_C_stage2: 0.34444444444444444
    embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831: 0.44074074074074077
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.5277777777777778
    granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0: 0.4388888888888889
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.5148148148148148
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.5111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.5092592592592593
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.4777777777777778
    qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0: 0.46296296296296297
    qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0: 0.2833333333333333
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.5074074074074074
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.42962962962962964
    ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff: 0.5518518518518518
  forward_selection_steps:
  - step: 1
    added_candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    accuracy: 0.5518518518518518
    log_loss: 2.5716356138271537
  - step: 2
    added_candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    accuracy: 0.5796296296296296
    log_loss: 1.323572680003492
  - step: 3
    added_candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    accuracy: 0.6018518518518519
    log_loss: 1.250228507537642
  optimization_summary: artifacts/predictions/prediction_v0_e805c0541629_summary.json
  forward_steps_artifact: artifacts/predictions/prediction_v0_e805c0541629_forward_steps.csv
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_e805c0541629
  prediction_manifest: artifacts/predictions/prediction_v0_e805c0541629_manifest.json
  validation_rows: 540
  optimizer_protocol: forward_group_cv_weighted_v1
- version: v10
  created_at_utc: '2026-11-08T23:04:10.740067+00:00'
  trigger_experiment_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
  candidate_roster_hash: f0b1c81936051a82dc985b5957c14f6b92efe814bcd0552927fd43d88872cbc0
  candidate_checkpoints:
  - model: catboost_stylometry
    candidate_id: catboost_two_stage_20261108T0850Z_C_stage2
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_two_stage_20261108T0850Z_C_stage2_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_two_stage/20261108T0850Z/C/stage2/checkpoint
    checkpoint_filename: checkpoint
    selected_epoch: null
    validation_balanced_accuracy: 0.34444444444444444
  - model: embeddinggemma2
    candidate_id: embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.4407407407407407
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_350m
    candidate_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.48518518518518516
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: qwen17
    candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.4629629629629629
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 2
    validation_balanced_accuracy: 0.5518518518518518
  selected_strategy: equal_probability_blend
  selected_comparison: previous_champion
  selected_members:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  validation_metric_scope: prompt_group_out_of_fold
  full_data_metrics_exploratory:
    accuracy: 0.6
    balanced_accuracy: 0.6000000000000001
    log_loss: 1.1764712402418678
  group_oof_metrics:
    equal_probability_blend:
      accuracy: 0.5907407407407408
      balanced_accuracy: 0.5907407407407407
      log_loss: 1.191491891656503
      ensemble_size: 5
    optimized_probability_blend:
      accuracy: 0.5907407407407408
      balanced_accuracy: 0.5907407407407407
      log_loss: 1.1761444701087513
      ensemble_size: 5
    all_candidate_weighted_blend:
      accuracy: 0.6018518518518519
      balanced_accuracy: 0.6018518518518517
      log_loss: 1.1183437123274387
      ensemble_size: 13
    previous_champion:
      accuracy: 0.6
      balanced_accuracy: 0.6000000000000001
      log_loss: 1.1764712402418678
      ensemble_size: 9
  selected_weights:
    catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0: 0.1111111111111111
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.1111111111111111
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.1111111111111111
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.1111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.1111111111111111
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.1111111111111111
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.1111111111111111
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.1111111111111111
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.1111111111111111
  individual_accuracy:
    catboost_two_stage_20261108T0850Z_C_stage2: 0.34444444444444444
    embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831: 0.44074074074074077
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.5277777777777778
    granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831: 0.48518518518518516
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.5148148148148148
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.5111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.5092592592592593
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.4777777777777778
    qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0: 0.46296296296296297
    qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0: 0.2833333333333333
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.5074074074074074
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.42962962962962964
    ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff: 0.5518518518518518
  forward_selection_steps:
  - step: 1
    added_candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    accuracy: 0.5518518518518518
    log_loss: 2.5716356138271537
  - step: 2
    added_candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    accuracy: 0.5796296296296296
    log_loss: 1.323572680003492
  - step: 3
    added_candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    accuracy: 0.6018518518518519
    log_loss: 1.250228507537642
  - step: 4
    added_candidate_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
    accuracy: 0.6055555555555555
    log_loss: 1.1784423117752338
  - step: 5
    added_candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    accuracy: 0.6111111111111112
    log_loss: 1.1750352454573842
  optimization_summary: artifacts/predictions/prediction_v0_0c2901b1a431_summary.json
  forward_steps_artifact: artifacts/predictions/prediction_v0_0c2901b1a431_forward_steps.csv
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_0c2901b1a431
  prediction_manifest: artifacts/predictions/prediction_v0_0c2901b1a431_manifest.json
  validation_rows: 540
  optimizer_protocol: forward_group_cv_weighted_v1
- version: v11
  created_at_utc: '2026-11-09T13:25:28.474413+00:00'
  trigger_experiment_id: granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831
  candidate_roster_hash: 25f88edb294c2e48a1ebc2627cb448312cc55c2bda05ee3dfcb666c2ee4b2275
  candidate_checkpoints:
  - model: catboost_stylometry
    candidate_id: catboost_two_stage_20261108T0850Z_C_stage2
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_two_stage_20261108T0850Z_C_stage2_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_two_stage/20261108T0850Z/C/stage2/checkpoint
    checkpoint_filename: checkpoint
    selected_epoch: null
    validation_balanced_accuracy: 0.34444444444444444
  - model: embeddinggemma2
    candidate_id: embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.4407407407407407
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_350m
    candidate_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.48518518518518516
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5277777777777778
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: qwen17
    candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.4629629629629629
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 2
    validation_balanced_accuracy: 0.5518518518518518
  selected_strategy: equal_probability_blend
  selected_comparison: previous_champion
  selected_members:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  validation_metric_scope: prompt_group_out_of_fold
  full_data_metrics_exploratory:
    accuracy: 0.6
    balanced_accuracy: 0.6000000000000001
    log_loss: 1.1764712402418678
  group_oof_metrics:
    equal_probability_blend:
      accuracy: 0.575925925925926
      balanced_accuracy: 0.575925925925926
      log_loss: 1.1895674440731279
      ensemble_size: 4
    optimized_probability_blend:
      accuracy: 0.5703703703703704
      balanced_accuracy: 0.5703703703703703
      log_loss: 1.1610521879370483
      ensemble_size: 4
    all_candidate_weighted_blend:
      accuracy: 0.5962962962962963
      balanced_accuracy: 0.5962962962962962
      log_loss: 1.1145385584954046
      ensemble_size: 13
    previous_champion:
      accuracy: 0.6
      balanced_accuracy: 0.6000000000000001
      log_loss: 1.1764712402418678
      ensemble_size: 9
  selected_weights:
    catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0: 0.1111111111111111
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.1111111111111111
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.1111111111111111
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.1111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.1111111111111111
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.1111111111111111
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.1111111111111111
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.1111111111111111
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.1111111111111111
  individual_accuracy:
    catboost_two_stage_20261108T0850Z_C_stage2: 0.34444444444444444
    embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831: 0.44074074074074077
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.5277777777777778
    granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831: 0.48518518518518516
    granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831: 0.5277777777777778
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.5111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.5092592592592593
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.4777777777777778
    qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0: 0.46296296296296297
    qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0: 0.2833333333333333
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.5074074074074074
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.42962962962962964
    ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff: 0.5518518518518518
  forward_selection_steps:
  - step: 1
    added_candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    accuracy: 0.5518518518518518
    log_loss: 2.5716356138271537
  - step: 2
    added_candidate_id: granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831
    accuracy: 0.5814814814814815
    log_loss: 1.5682877351256517
  - step: 3
    added_candidate_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
    accuracy: 0.5981481481481481
    log_loss: 1.2608491990446662
  - step: 4
    added_candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    accuracy: 0.6018518518518519
    log_loss: 1.2131599390362307
  optimization_summary: artifacts/predictions/prediction_v0_7df6e8a23dbb_summary.json
  forward_steps_artifact: artifacts/predictions/prediction_v0_7df6e8a23dbb_forward_steps.csv
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_7df6e8a23dbb
  prediction_manifest: artifacts/predictions/prediction_v0_7df6e8a23dbb_manifest.json
  validation_rows: 540
  optimizer_protocol: forward_group_cv_weighted_v1
- version: v12
  created_at_utc: '2026-11-09T19:37:44.966675+00:00'
  trigger_experiment_id: llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81
  candidate_roster_hash: fd0d8fe447aca84a7d533b0fd4447ea9d49bdc4f680dc488dd18466a82bbab4c
  candidate_checkpoints:
  - model: catboost_stylometry
    candidate_id: catboost_two_stage_20261108T0850Z_C_stage2
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_two_stage_20261108T0850Z_C_stage2_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_two_stage/20261108T0850Z/C/stage2/checkpoint
    checkpoint_filename: checkpoint
    selected_epoch: null
    validation_balanced_accuracy: 0.34444444444444444
  - model: embeddinggemma2
    candidate_id: embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.4407407407407407
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_350m
    candidate_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.48518518518518516
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5277777777777778
  - model: llama32_3b
    candidate_id: llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 2
    validation_balanced_accuracy: 0.5425925925925926
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: qwen17
    candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.4629629629629629
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 2
    validation_balanced_accuracy: 0.5518518518518518
  selected_strategy: equal_probability_blend
  selected_comparison: previous_champion
  selected_members:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  validation_metric_scope: prompt_group_out_of_fold
  full_data_metrics_exploratory:
    accuracy: 0.6
    balanced_accuracy: 0.6000000000000001
    log_loss: 1.1764712402418678
  group_oof_metrics:
    equal_probability_blend:
      accuracy: 0.575925925925926
      balanced_accuracy: 0.5759259259259258
      log_loss: 1.1747678923333091
      ensemble_size: 4
    optimized_probability_blend:
      accuracy: 0.5611111111111111
      balanced_accuracy: 0.5611111111111111
      log_loss: 1.1392995150980785
      ensemble_size: 4
    all_candidate_weighted_blend:
      accuracy: 0.5944444444444444
      balanced_accuracy: 0.5944444444444444
      log_loss: 1.0923263041824014
      ensemble_size: 13
    previous_champion:
      accuracy: 0.6
      balanced_accuracy: 0.6000000000000001
      log_loss: 1.1764712402418678
      ensemble_size: 9
  selected_weights:
    catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0: 0.1111111111111111
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.1111111111111111
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.1111111111111111
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.1111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.1111111111111111
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.1111111111111111
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.1111111111111111
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.1111111111111111
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.1111111111111111
  individual_accuracy:
    catboost_two_stage_20261108T0850Z_C_stage2: 0.34444444444444444
    embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831: 0.44074074074074077
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.5277777777777778
    granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831: 0.48518518518518516
    granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831: 0.5277777777777778
    llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81: 0.5425925925925926
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.5092592592592593
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.4777777777777778
    qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0: 0.46296296296296297
    qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0: 0.2833333333333333
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.5074074074074074
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.42962962962962964
    ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff: 0.5518518518518518
  forward_selection_steps:
  - step: 1
    added_candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    accuracy: 0.5518518518518518
    log_loss: 2.5716356138271537
  - step: 2
    added_candidate_id: granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831
    accuracy: 0.5814814814814815
    log_loss: 1.5682877351256517
  - step: 3
    added_candidate_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
    accuracy: 0.5981481481481481
    log_loss: 1.2608491990446662
  - step: 4
    added_candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    accuracy: 0.6018518518518519
    log_loss: 1.2131599390362307
  optimization_summary: artifacts/predictions/prediction_v0_10a575dc33f5_summary.json
  forward_steps_artifact: artifacts/predictions/prediction_v0_10a575dc33f5_forward_steps.csv
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_10a575dc33f5
  prediction_manifest: artifacts/predictions/prediction_v0_10a575dc33f5_manifest.json
  validation_rows: 540
  optimizer_protocol: forward_group_cv_weighted_v1
- version: v13
  created_at_utc: '2026-11-10T03:22:49.642293+00:00'
  trigger_experiment_id: ministral3_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg4619b410_code279cc81
  candidate_roster_hash: 8a9aa4a86de47cb5df86f054819e17a0fb1ecb85afb74d147032d53378dc1020
  candidate_checkpoints:
  - model: catboost_stylometry
    candidate_id: catboost_two_stage_20261108T0850Z_C_stage2
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_two_stage_20261108T0850Z_C_stage2_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_two_stage/20261108T0850Z/C/stage2/checkpoint
    checkpoint_filename: checkpoint
    selected_epoch: null
    validation_balanced_accuracy: 0.34444444444444444
  - model: embeddinggemma2
    candidate_id: embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831/checkpoints/embeddinggemma2_foldvalidation
    checkpoint_filename: embeddinggemma2_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.4407407407407407
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_350m
    candidate_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831/checkpoints/granite4_350m_foldvalidation
    checkpoint_filename: granite4_350m_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.48518518518518516
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5277777777777778
  - model: llama32_3b
    candidate_id: llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 2
    validation_balanced_accuracy: 0.5425925925925926
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg4619b410_code279cc81
    training_dataset: v1_all
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg4619b410_code279cc81_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg4619b410_code279cc81/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5203703703703703
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: qwen17
    candidate_id: qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0/checkpoints/qwen17_foldvalidation
    checkpoint_filename: qwen17_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.4629629629629629
  - model: qwen4
    candidate_id: qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0/checkpoints/qwen4_foldvalidation
    checkpoint_filename: qwen4_foldvalidation
    selected_epoch: 9
    validation_balanced_accuracy: 0.2833333333333333
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    training_dataset: v0
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 2
    validation_balanced_accuracy: 0.5518518518518518
  selected_strategy: equal_probability_blend
  selected_comparison: previous_champion
  selected_members:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  validation_metric_scope: prompt_group_out_of_fold
  full_data_metrics_exploratory:
    accuracy: 0.6
    balanced_accuracy: 0.6000000000000001
    log_loss: 1.1764712402418678
  group_oof_metrics:
    equal_probability_blend:
      accuracy: 0.5703703703703704
      balanced_accuracy: 0.5703703703703703
      log_loss: 1.1684889543446917
      ensemble_size: 4
    optimized_probability_blend:
      accuracy: 0.5648148148148148
      balanced_accuracy: 0.5648148148148148
      log_loss: 1.1219958319004777
      ensemble_size: 4
    all_candidate_weighted_blend:
      accuracy: 0.6018518518518519
      balanced_accuracy: 0.6018518518518517
      log_loss: 1.0811889076992303
      ensemble_size: 13
    previous_champion:
      accuracy: 0.6
      balanced_accuracy: 0.6000000000000001
      log_loss: 1.1764712402418678
      ensemble_size: 9
  selected_weights:
    catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0: 0.1111111111111111
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.1111111111111111
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.1111111111111111
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.1111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.1111111111111111
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.1111111111111111
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.1111111111111111
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.1111111111111111
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.1111111111111111
  individual_accuracy:
    catboost_two_stage_20261108T0850Z_C_stage2: 0.34444444444444444
    embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831: 0.44074074074074077
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.5277777777777778
    granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831: 0.48518518518518516
    granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831: 0.5277777777777778
    llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81: 0.5425925925925926
    ministral3_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg4619b410_code279cc81: 0.5203703703703704
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.4777777777777778
    qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0: 0.46296296296296297
    qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0: 0.2833333333333333
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.5074074074074074
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.42962962962962964
    ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff: 0.5518518518518518
  forward_selection_steps:
  - step: 1
    added_candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    accuracy: 0.5518518518518518
    log_loss: 2.5716356138271537
  - step: 2
    added_candidate_id: granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831
    accuracy: 0.5814814814814815
    log_loss: 1.5682877351256517
  - step: 3
    added_candidate_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
    accuracy: 0.5981481481481481
    log_loss: 1.2608491990446662
  - step: 4
    added_candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    accuracy: 0.6018518518518519
    log_loss: 1.2131599390362307
  optimization_summary: artifacts/predictions/prediction_v0_223f172caf53_summary.json
  forward_steps_artifact: artifacts/predictions/prediction_v0_223f172caf53_forward_steps.csv
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_223f172caf53
  prediction_manifest: artifacts/predictions/prediction_v0_223f172caf53_manifest.json
  validation_rows: 540
  optimizer_protocol: forward_group_cv_weighted_v1
- version: v14
  created_at_utc: '2026-11-10T10:25:47.162615+00:00'
  trigger_experiment_id: qwen17_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgd07c657b_code279cc81
  candidate_roster_hash: 8294d46b142142d173c4c91083ff03d98311eb28ea34c54dc1a4680ffce98564
  candidate_checkpoints: *id037
  selected_strategy: equal_probability_blend
  selected_comparison: previous_champion
  selected_members:
  - model: catboost_stylometry
    candidate_id: catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0/checkpoints/catboost_stylometry_foldvalidation
    checkpoint_filename: catboost_stylometry_foldvalidation
    selected_epoch: null
    validation_balanced_accuracy: 0.3444444444444444
  - model: gemma4_e2b
    candidate_id: gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0/checkpoints/gemma4_e2b_foldvalidation
    checkpoint_filename: gemma4_e2b_foldvalidation
    selected_epoch: 7
    validation_balanced_accuracy: 0.5277777777777778
  - model: granite4_h_micro
    candidate_id: granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0/checkpoints/granite4_h_micro_foldvalidation
    checkpoint_filename: granite4_h_micro_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5148148148148148
  - model: llama32_3b
    candidate_id: llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0/checkpoints/llama32_3b_foldvalidation
    checkpoint_filename: llama32_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5111111111111111
  - model: ministral3_3b
    candidate_id: ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0
    training_dataset: v1.1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0/checkpoints/ministral3_3b_foldvalidation
    checkpoint_filename: ministral3_3b_foldvalidation
    selected_epoch: 3
    validation_balanced_accuracy: 0.5092592592592593
  - model: qwen06
    candidate_id: qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0/checkpoints/qwen06_foldvalidation
    checkpoint_filename: qwen06_foldvalidation
    selected_epoch: 6
    validation_balanced_accuracy: 0.47777777777777775
  - model: smollm3_3b
    candidate_id: smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0/checkpoints/smollm3_3b_foldvalidation
    checkpoint_filename: smollm3_3b_foldvalidation
    selected_epoch: 5
    validation_balanced_accuracy: 0.5074074074074074
  - model: svc
    candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    training_dataset: v1
    source_manifest: /home/zodrak/nlm/artifacts/predictions/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0/checkpoints/svc_foldvalidation.joblib
    checkpoint_filename: svc_foldvalidation.joblib
    selected_epoch: null
    validation_balanced_accuracy: 0.4166666666666667
  - model: ultrabert380
    candidate_id: ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0
    training_dataset: v1.5
    source_manifest: /home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json
    checkpoint_path: /home/zodrak/nlm/artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued
    checkpoint_filename: ultrabert380_continued
    selected_epoch: 5
    validation_balanced_accuracy: 0.5314814814814814
  predicted_balanced_accuracy: 0.6000000000000001
  validation_accuracy: 0.6
  validation_metric_scope: prompt_group_out_of_fold
  full_data_metrics_exploratory:
    accuracy: 0.6
    balanced_accuracy: 0.6000000000000001
    log_loss: 1.1764712402418678
  group_oof_metrics:
    equal_probability_blend:
      accuracy: 0.562962962962963
      balanced_accuracy: 0.5629629629629629
      log_loss: 1.1690176310961486
      ensemble_size: 4
    optimized_probability_blend:
      accuracy: 0.5611111111111111
      balanced_accuracy: 0.5611111111111111
      log_loss: 1.125711500336446
      ensemble_size: 4
    all_candidate_weighted_blend:
      accuracy: 0.6
      balanced_accuracy: 0.6
      log_loss: 1.0772941820614124
      ensemble_size: 13
    previous_champion:
      accuracy: 0.6
      balanced_accuracy: 0.6000000000000001
      log_loss: 1.1764712402418678
      ensemble_size: 9
  selected_weights:
    catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0: 0.1111111111111111
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.1111111111111111
    granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0: 0.1111111111111111
    llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0: 0.1111111111111111
    ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0: 0.1111111111111111
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.1111111111111111
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.1111111111111111
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.1111111111111111
    ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0: 0.1111111111111111
  individual_accuracy:
    catboost_two_stage_20261108T0850Z_C_stage2: 0.34444444444444444
    embeddinggemma2_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg55aefb6c_coded9e8831: 0.44074074074074077
    gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0: 0.5277777777777778
    granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831: 0.48518518518518516
    granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831: 0.5277777777777778
    llama32_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgfc0a592c_code279cc81: 0.5425925925925926
    ministral3_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg4619b410_code279cc81: 0.5203703703703704
    qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0: 0.4777777777777778
    qwen17_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgd07c657b_code279cc81: 0.524074074074074
    qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0: 0.2833333333333333
    smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0: 0.5074074074074074
    svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0: 0.42962962962962964
    ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff: 0.5518518518518518
  forward_selection_steps:
  - step: 1
    added_candidate_id: ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff
    accuracy: 0.5518518518518518
    log_loss: 2.5716356138271537
  - step: 2
    added_candidate_id: granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831
    accuracy: 0.5814814814814815
    log_loss: 1.5682877351256517
  - step: 3
    added_candidate_id: granite4_350m_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg419401e9_coded9e8831
    accuracy: 0.5981481481481481
    log_loss: 1.2608491990446662
  - step: 4
    added_candidate_id: svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0
    accuracy: 0.6018518518518519
    log_loss: 1.2131599390362307
  optimization_summary: artifacts/predictions/prediction_v0_9b12fd46c588_summary.json
  forward_steps_artifact: artifacts/predictions/prediction_v0_9b12fd46c588_forward_steps.csv
  submitted_balanced_accuracy: 0.0
  prediction_experiment_id: prediction_v0_9b12fd46c588
  prediction_manifest: artifacts/predictions/prediction_v0_9b12fd46c588_manifest.json
  validation_rows: 540
  optimizer_protocol: forward_group_cv_weighted_v1
```
<!-- ensemble-registry:end -->
