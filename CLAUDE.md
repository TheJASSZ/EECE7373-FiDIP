# Project rules — EECE 7373 FiDIP hands-on (read before acting)

- Read `SKILL.md` before any cluster action. Always reach the cluster via `ssh -o BatchMode=yes -o ConnectTimeout=15 explorer '...'`, and wrap slow commands in `timeout`.
- No compute on login nodes. Use `sbatch` (templates in `scripts/slurm/`). Smoke-test on `sharing` first.
- **Never select models, epochs or hyperparameters using Test100 (validate100).** Select on Val400. Evaluate Test100 only for configs listed in `results/PREREGISTRATION.md`.
- Every reported metric must come from `eval/eval_coco.py` applied to a prediction JSON.
- Keep data and checkpoints on `/scratch/purohitham.v/fidip`. The Mac disk is nearly full, so sync only small files (metrics, preds, figures).
- Log as you go:
  - Detail (commands, job IDs, errors, fixes) goes in `notebook/<stream>.md`.
  - The orchestrator merges decisions into `AGENT.md` and inventory into `MANIFEST.md`.
  - Register runs in `results/registry.csv`.
- Edit vendored upstream code (`third_party/Infant-Pose-Estimation`) only in small, separately described patches.
- Secrets live in `.env` (gitignored). Never print or commit them.
- Commit messages: **no `Co-Authored-By` trailer and no mention of Claude as co-author** (the human's instruction, 2026-11-05). The report/README author is the student only. The AI-tool-use section describes the agent as a tool.
- Timestamps in notebooks: always take them from `date "+%Y-%m-%d %H:%M %Z"`. Never estimate.
