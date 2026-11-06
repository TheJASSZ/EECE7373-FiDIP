#!/usr/bin/env bash
# IMPROVE stream: sync the repo to $R/repo (via scripts/sync_to_cluster.sh), then write the git
# provenance of the synced tree to $R/repo/GIT_INFO_mmpose.json. The cluster copy has no .git,
# so run.json reads commit/tree/dirty from this file. Tree SHAs survive commit-message rewrites.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
bash scripts/sync_to_cluster.sh
HEAD_SHA=$(git rev-parse HEAD)
TREE_SHA=$(git rev-parse 'HEAD^{tree}')
DIRTY_FILES=$(git status --porcelain --untracked-files=all -- mmpose_exps scripts eval envs | awk '{print $2}' | tr '\n' ' ')
if [ -n "$DIRTY_FILES" ]; then DIRTY=true; else DIRTY=false; fi
SYNCED_AT=$(date "+%Y-%m-%dT%H:%M:%S%z")
python3 - "$HEAD_SHA" "$TREE_SHA" "$DIRTY" "$SYNCED_AT" "$DIRTY_FILES" <<'EOF' | ssh -o BatchMode=yes -o ConnectTimeout=15 explorer 'cat > /scratch/purohitham.v/fidip/repo/GIT_INFO_mmpose.json'
import json, sys
head, tree, dirty, ts, files = sys.argv[1:6]
print(json.dumps({"git_head": head, "git_tree": tree, "dirty": dirty == "true",
                  "dirty_files": files.split(), "synced_at": ts}, indent=1))
EOF
echo "git info: head=$HEAD_SHA tree=$TREE_SHA dirty=$DIRTY"
