# Pre-registration of final Test100 evaluations

Rule: a configuration may be evaluated on **Test100** only after it is listed here, with its selection evidence on **Val400**, in a commit made *before* that Test100 evaluation runs. The checkpoint selection rule is fixed here too (e.g. "final epoch" or "best Val400 AP epoch").

Exception: the upstream *released* checkpoints, and our official-recipe retrains, are reproduction targets rather than tuned candidates, so they are evaluated on Test100 directly.

| # | Registered at | Config / RUN_IDs | Selection evidence (Val400) | Checkpoint rule | Test100 result (filled after) |
|---|---|---|---|---|---|
| repro-1 | 2026-11-06 01:10 EST | R010-R012 `fidip-w48_lambda5e-4_s{1,2,3}` and R013-R015 `fidip-w48_lambda0_s{1,2,3}`: official recipe (upstream infant yaml + COCO DarkPose init `TRAIN.CHECKPOINT` + PRE_EPOCH 10 on the shipped train_pre_infant; λ = 5e-4 vs 0; per-epoch validation on Val400) | reproduction runs (exception clause); Val400 only picks the epoch | report BOTH: (a) epoch with best Val400 AP (`model_best.pth`), (b) final epoch 20 (`final_state.pth`); mean ± sd over 3 seeds | eval_coco paper protocol (fixed): λ5e-4 best 93.55±0.42 (95.11±0.46), final 93.38±0.48 (94.91±0.47); λ0 best 93.59±0.13 (95.08±0.23), final 93.66±0.10 (95.16±0.16). Filled 2026-11-06 07:30 |
| repro-2 | 2026-11-06 01:10 EST | R016-R018 λ sweep {1e-4, 2e-3, 1e-2} s1; R019 paper recipe (LR 1e-3, 100 ep, batch 64, λ 5e-4) s1; R020 README-literal labels (syrip_for_train train json) s1 | exploratory | **Val400 only, no Test100** | n/a |
| repro-3 | 2026-11-06 07:23 EST | **Plain fine-tune control (the paper's "DarkPose + Finetune" 92.7 row):** R021-R023, seeds s1–s3. Shipped infant yaml with `PRE_EPOCH 0` and `TRAIN.LAMBDA 0.0` (no classifier pre-stage, no adversarial term), COCO DarkPose init via `TRAIN.CHECKPOINT`, per-epoch validation on Val400 | reproduction run (exception clause). Motivation: R013–R015 (λ = 0) still include the PRE_EPOCH-10 classifier pre-stage, which updates BatchNorm statistics | same as repro-1: report BOTH the best-Val400 epoch and the final epoch; mean ± sd over 3 seeds | (filled after) |
| imp-1 | 2026-11-06 07:06 EST | **Arm A (control):** HRNet-W48 DARK 384×288, `mmpose_exps/configs/syrip_hrnet-w48_dark-384x288.py` (COCO init, lr 1e-4, batch 32, 30 epochs, step 24/29, 1:1 real:synthetic). **5 fresh seeds s2–s6 = R131-R135** | see *Selection evidence* below | per seed: the epoch with the best Val400 coco/AP (`save_best`, evaluated every epoch). SHA256 is in run.json and verified by `final_test_eval_mmpose.sbatch` | (filled after) |
| imp-2 | 2026-11-06 07:06 EST | **Arm B (treatment):** ViTPose-H 256×192 UDP, `mmpose_exps/configs/syrip_vitpose-h_udp-256x192.py` (COCO init, same schedule, batch and data as arm A). **5 fresh seeds s2–s6 = R136-R140** | see *Selection evidence* below | same as imp-1 | (filled after) |
| imp-3 | 2026-11-06 07:06 EST | **Secondary:** a 5-seed ensemble of arm B = R141 (ensemble), built by `eval/ensemble_preds.py` over the Test100 predictions of R136-R140: per-instance mean of the flip-tested keypoint coordinates; instance score = mean of member scores | — | inherits imp-2 | (filled after) |
| imp-4 | 2026-11-06 07:06 EST | **Supplementary, reported separately and NOT in the primary mean:** screening seed s1 of each arm, which was used for configuration selection: R101 (arm A s1), R105 (arm B s1) | — | same as imp-1 | (filled after) |

## Registration imp-1…imp-4: plan and decision rule (committed 2026-11-06 07:06 EST, before any Test100 evaluation of arms A/B)

### Selection evidence (Val400, 397 images; `eval_coco`; seed s1; paper = fixed on Val400)
- Configs compared on Val400 only:

  - HRNet-W48 (R101): 91.52; HRNet lr 2e-4 / 5e-5: 91.43 / 91.69
  - ViT-L 256 (R102): 92.81; ViT-L 256 lr 2e-4 / 5e-5: 92.58 / 92.60
  - ViT-L 384 (R103): 92.77
  - **ViT-H 256 (R105): 93.17**
  - ViT-H 384 (R104): 93.27
  - (Written as bullets, not a table, on purpose: `scripts/prereg.py` treats every *table row* in this file as a Test100 registration.)

- For reference: zero-shot COCO models score 86.3–88.8, the released FiDIP 90.4, and our FiDIP retrains 90.4–90.6.
- The LR sweeps are flat (spread 0.15–0.25), so single-seed noise is about ±0.2.
  - ViT-H beats ViT-L by about +0.4, which is 2–3× the noise.
  - 384 vs 256 is within noise (+0.10 for H).
- **ViT-H 256 was chosen over ViT-H 384.** The gap is within noise, and 384 needs about 2× the compute and a 43–46 GB peak, so only the scarce A100-80 nodes can run it.
- **The arm A recipe is identical to arm B's** (same schedule, batch, data and LR), so the A-vs-B contrast isolates the backbone. The flat LR sweep for HRNet shows the control is not handicapped by its LR.
- **Fresh seeds s2–s6 are used for both arms**, so the screening seed that won the selection (s1) cannot inflate the primary estimate (winner's curse). s1 is reported separately (imp-4).

### Evaluation conditions (identical for every arm)
- Test100 = official validate100, exactly as shipped (100 images).
- GT person boxes and flip test.
- `eval/eval_coco.py`: pycocotools, 17 COCO sigmas, the shared rescoring rule.
- **Headline:** the `paper` protocol (annotation ids as shipped, comparable with the published 93.6).
- **Sensitivity checks:** the `fixed` protocol (D10), and Test100-clean (95 images, D16).

### Primary hypothesis H1 (decision rule fixed now)
The bonus improvement claim holds only if **both** of these are true:
1. The mean Test100 AP (paper protocol) over the 5 fresh seeds of arm B is greater than the published DarkPose+FiDIP **93.6**.
2. The paired-bootstrap 95% CI of ΔAP (arm B, seed-averaged, minus the released FiDIP checkpoint) excludes 0 under **both** the paper and fixed protocols (D13), after Holm correction across C1–C4.

### Pre-declared contrasts
Method: paired bootstrap over the 100 Test100 images, 10,000 resamples, RNG seed 0, seed-averaged AP per resample, Holm correction across C1–C4.

| Contrast | Comparison | Role |
|---|---|---|
| **C1** | arm B vs released FiDIP checkpoint (the R001 predictions) | primary |
| **C2** | arm B vs arm A | the controlled backbone comparison; same recipe |
| **C3** | arm B vs our FiDIP retrains R010–R012 (best-Val400 checkpoints, seed-averaged) | |
| **C4** | the arm B ensemble (imp-3) vs released FiDIP | secondary |

- Also reported, outside the bonus claim: arm A vs released FiDIP, and the reproduction DA ablation (λ 5e-4 vs λ 0, R010–R012 vs R013–R015).

### If H1 fails
- We report "no verified improvement" with every number, CI and attempt. The rubric explicitly allows full credit without a win.
- No further configurations are tried against Test100 afterwards.

### Erratum (committed after registration commit 72f9449, before any Test100 evaluation of MMPose runs)
- In 72f9449 the selection evidence above was a markdown *table*, and the gate parser (`scripts/prereg.py`) reads every table row as a registration. That **unintentionally made screening runs R102, R103 and R104 eligible for Test100**.
- The same commit also wrote the ensemble ID so that the parser did not recognize it.
- This commit turns the evidence into bullets and fixes the ensemble token. Registered for Test100 are exactly: the released checkpoints, R010–R015 (repro-1), R101 and R105 (supplementary), R131–R140, and R141 (ensemble).
- Checked: no Test100 predictions or metrics exist for any unregistered run.
