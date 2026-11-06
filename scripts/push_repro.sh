#!/usr/bin/env bash
# Push the repo to the cluster (scripts/sync_to_cluster.sh, with retries) and record which git state was
# pushed in $R/repo/.repo_version.json (the cluster copy has no .git). run.json files copy it.
#   head / tree  : git rev-parse HEAD and HEAD^{tree} (tree SHAs survive commit-message rewrites)
#   dirty_any    : uncommitted changes anywhere in the working tree that was pushed
#   dirty_repro  : uncommitted changes under the paths the repro stream runs (third_party, scripts, demo)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
for i in 1 2 3 4 5; do
  if bash scripts/sync_to_cluster.sh; then break; fi
  echo "sync attempt $i failed; retrying in 10 s" >&2; sleep 10
  [[ $i == 5 ]] && exit 1
done
head=$(git rev-parse HEAD); tree=$(git rev-parse 'HEAD^{tree}')
dirty_any=$([[ -n "$(git status --porcelain)" ]] && echo true || echo false)
dirty_repro=$(git status --porcelain -- third_party scripts demo | sed 's/"/\\"/g' | awk '{printf "%s\"%s\"", (NR>1?",":""), $0}')
printf '{"head": "%s", "tree": "%s", "dirty_any": %s, "dirty_repro_paths": [%s], "pushed_at": "%s"}\n' \
  "$head" "$tree" "$dirty_any" "$dirty_repro" "$(date '+%Y-%m-%dT%H:%M:%S%z')" \
| ssh -o BatchMode=yes -o ConnectTimeout=15 explorer 'cat > /scratch/purohitham.v/fidip/repo/.repo_version.json'
echo "recorded HEAD $head tree $tree dirty_any=$dirty_any"
