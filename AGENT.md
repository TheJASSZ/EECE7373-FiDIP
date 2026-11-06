# AGENT.md — running lab notebook

Claude Code (model: Claude Opus 5.5) keeps this notebook live, working alongside the student (Vishnu Purohitham).
- Entries are chronological, with the newest at the bottom.
- This file records what the human asked, what the agent tried, what failed, and why the direction changed.
- Per-stream detail (commands, job IDs, stack traces) lives in `notebook/<stream>.md`.
- Inventory and provenance live in `MANIFEST.md`. Cluster how-to lives in `SKILL.md`.

## Decision index
| # | When | Decision | Why |
|---|---|---|---|
| D1 | 11-05 19:45 | **Use Val400 (validate500 \ validate100, 400 real images) for all model and epoch selection.** Test100 (validate100) is used only for pre-registered final evals. | Test100 is the paper's reporting set. The upstream training loop appears to pick `model_best` on the evaluation set, which amounts to test-set selection. The bonus rules forbid test-set tuning. Val400 is disjoint from Test100 and leaves the training data identical to the paper's (200 real + 1000 synthetic). |
| D2 | 11-05 19:45 | **Every number comes from one evaluator** (`eval/eval_coco.py`: pycocotools, standard 17-keypoint COCO sigmas, GT boxes, one rescoring rule) applied to prediction JSONs. | The rubric asks for "comparable evaluation conditions". Two frameworks (upstream FiDIP, MMPose) must not differ in their metric code. |
| D3 | 11-05 19:40 | **Data, environments and checkpoints stay on Explorer `/scratch`.** Only code, docs, metrics and figures sync to the Mac. | The Mac has 7.6 GB free (97% full). The cluster has H200/A100 GPUs. |
| D4 | 11-05 19:48 | **Run four parallel agent streams** (repro, improve, analysis, report), with this orchestrator owning AGENT/MANIFEST/SKILL. | The human asked to "deploy as many agents quickly". The streams are independent once data is downloaded. |

---

## Log

### 2026-11-05 ~19:25–19:52 EST — Session 1: kickoff and planning (plan mode, read-only)
**Human asked:**
- First message: help completing the FiDIP hands-on exercise with "maximum points + bonus". The grading breakdown was pasted (50 pts: implementation and reproducibility 20, results and evaluation 15, report and documentation 15; up to +5 bonus for verified improvement). The human also asked to "create a folder in documents and start working".
- Then: "Wait". The session switched to plan mode.
- Shared the course slides ("EECE 7373 — Utilizing the Explorer Cluster", incl. required SKILL.md / AGENT.md / MANIFEST.md) with "use NEU HPC if you need".
- Follow-ups: "Please continue on the work", "Deploy as many agents quickly to finish off the work in efficient way possible".

**Agent did (read-only exploration):**
- **Local machine:** Apple M4, 16 GB RAM, **7.6 GB free disk**. conda, pandoc, gh and git are present. No LaTeX.
- **Course material search:** Spotlight found no STUDENT_GUIDE.md, SyRIP or FiDIP files locally. `~/.ssh/config` already has `Host explorer`.
- **First SSH probe:** hung more than 60 s. The cause was a slow `du ~` / `module avail` on the login node (stdout was buffered behind `head`). I stopped it.
- **Verbose SSH:** showed public-key auth succeeding in 3.2 s, so **passwordless SSH works**.
- **Lighter probe (with `timeout` wrappers):**
  - Slurm accounts: `cs5100.202610`, `cs5130.202630` (default).
  - Partitions: `gpu` (8 h; H200/A100/V100), `sharing` (1 h; A100/H100/L40…), `short` (2 d).
  - Modules: `miniconda3/25.9.1`, `cuda/12.1.1…12.8.0`.
  - `/scratch/purohitham.v` exists.
  - **`/projects/aclab` → permission denied**, so the lab's copy of the data can't be reused.
  - No TeX on the cluster.
- **Official repo README** (github.com/ostadabbas/Infant-Pose-Estimation):
  - Published Test100 numbers: DarkPose+FiDIP HRNet-W48 384×288 **AP 93.6** (AR 94.6). SimpleBaseline+FiDIP R-50 91.1. MobileNet+FiDIP 79.2. DarkPose fine-tune without DA 92.7.
  - Environment is now Python 3.12 / CUDA 12.1 (`fidip_env.yml`).
  - Checkpoints are on Google Drive.
- **SyRIP dataset page:** `SyRIP.zip` 313 MB, `syrip_for_train.zip` 608 MB. The README says train = 200 real + 1000 synthetic, validate500 = 500 real, and **validate100 (= Test100) ⊂ validate500**.
- **Background Explore agents launched:**
  1. A code audit of the upstream repo: env, configs, training flow, what `model_best` is selected on, needed compat patches.
  2. A literature/engineering scan of what has beaten FiDIP on SyRIP and how (ViTPose / MMPose specifics).

**What failed / surprises:**
- The hung SSH probe described above.
- No lab-share access.
- The Mac disk is nearly full.
- `train_pre_infant` (1904 COCO val2017 + 2000 SURREAL adults, used by FiDIP's domain-classifier stage) is "available on request" per the README. That is a possible reproduction blocker; it is being checked.

**Decisions:** D1–D4 above.

**Plan:** written in plan mode and **approved by the human at ~19:50 EST**. Summary:
- Reproduce: released checkpoints, then an official retrain × 3 seeds.
- Improve: controlled backbone swap (HRNet-W48 vs ViTPose) under one MMPose recipe, plus data/augmentation/TTA ablations, all selected on Val400.
- Then pre-register, run the final Test100 eval with paired-bootstrap CIs, write the LaTeX report, and have a fresh-context reviewer grade against the rubric.

### 2026-11-05 19:52 EST — Step 0: scaffolding
- Created the local git repo `~/Documents/EECE7373-FiDIP` with the planned layout.
- Created the cluster tree `/scratch/purohitham.v/fidip/{repo,data,models,envs,runs,slurm_logs,logs}` and `~/fidip_artifacts`.
- Started the SyRIP download + unzip as a background job on the login node (pid 687507, explorer-02, 19:54 EST). Log: `$R/data/downloads/download.log`. Markers: `DOWNLOAD_DONE`, `UNZIP_DONE`. Checksums go to `SHA256SUMS`.
- Wrote the first versions of SKILL.md (connection facts, partitions, ops, guardrails), MANIFEST.md (inventory skeleton) and CLAUDE.md (project rules every agent loads).
