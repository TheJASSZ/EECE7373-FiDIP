# Pre-registration of final Test100 evaluations

Rule: a configuration may be evaluated on **Test100** only after it is listed here, with its selection evidence on **Val400**, in a commit made *before* that Test100 evaluation runs. The checkpoint selection rule is fixed here too (e.g. "final epoch" or "best Val400 AP epoch").

Exception: the upstream *released* checkpoints, and our official-recipe retrains, are reproduction targets rather than tuned candidates, so they are evaluated on Test100 directly.

| # | Registered at | Config / RUN_IDs | Selection evidence (Val400) | Checkpoint rule | Test100 result (filled after) |
|---|---|---|---|---|---|
| repro-1 | 2026-11-06 01:10 EST | R010-R012 `fidip-w48_lambda5e-4_s{1,2,3}` and R013-R015 `fidip-w48_lambda0_s{1,2,3}`: official recipe (upstream infant yaml + COCO DarkPose init `TRAIN.CHECKPOINT` + PRE_EPOCH 10 on the shipped train_pre_infant; λ = 5e-4 vs 0; per-epoch validation on Val400) | reproduction runs (exception clause); Val400 only picks the epoch | report BOTH: (a) epoch with best Val400 AP (`model_best.pth`), (b) final epoch 20 (`final_state.pth`); mean ± sd over 3 seeds | (filled after) |
| repro-2 | 2026-11-06 01:10 EST | R016-R018 λ sweep {1e-4, 2e-3, 1e-2} s1; R019 paper recipe (LR 1e-3, 100 ep, batch 64, λ 5e-4) s1; R020 README-literal labels (syrip_for_train train json) s1 | exploratory | **Val400 only, no Test100** | n/a |
