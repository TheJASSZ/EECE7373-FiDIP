#!/usr/bin/env bash
# SyRIP data preparation for the FiDIP reproduction (EECE 7373).
#
# Stages (run in the order given; default: verify unzip verifyraw layout splits):
#   download : fetch SyRIP.zip, syrip_for_train.zip, README.md into $R/data/downloads (skips files that exist)
#   verify   : check SHA256 of the downloads against $R/data/downloads/SHA256SUMS
#   unzip    : extract both zips into $R/data/raw/   ** compute node only: login nodes kill unzip **
#   layout   : build $R/data/syrip in the upstream FiDIP layout (symlinks only, no copies)
#   verifyraw: every zip member exists on disk with the same size + CRC32, no extra files, md5 sample
#   neardups : (optional, needs the fidip env) perceptual near-duplicate scan -> splits/val400_exclude_near_dups.txt
#   splits   : write framework-neutral COCO GT files to $R/data/splits + verification report
#              (scripts/make_splits.py), then touch $R/data/SPLITS_READY
#
# Usage (on the cluster, from $R/repo):
#   sbatch scripts/slurm/get_data.sbatch                 # all default stages, CPU job on `short`
#   bash scripts/get_data.sh layout splits               # cheap stages only (symlinks + json)
set -euo pipefail

R=${R:-/scratch/purohitham.v/fidip}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DL=$R/data/downloads
RAW=$R/data/raw
SY=$R/data/syrip
SPLITS=$R/data/splits
BASE_URL=https://coe.northeastern.edu/Research/AClab/SyRIP
PY=${PY:-python3}

log() { echo "[get_data $(date '+%F %T')] $*"; }

stage_download() {
  mkdir -p "$DL"
  for f in SyRIP.zip syrip_for_train.zip README.md; do
    if [[ -s "$DL/$f" ]]; then log "have $f"; continue; fi
    log "downloading $BASE_URL/$f"
    curl -fL --retry 3 -C - -o "$DL/$f.part" "$BASE_URL/$f" && mv "$DL/$f.part" "$DL/$f"
  done
  if [[ ! -s "$DL/SHA256SUMS" ]]; then
    (cd "$DL" && sha256sum SyRIP.zip syrip_for_train.zip README.md > SHA256SUMS)
    log "wrote $DL/SHA256SUMS (first download: these become the reference hashes)"
  fi
}

stage_verify() {
  (cd "$DL" && sha256sum -c SHA256SUMS)
}

stage_unzip() {
  # Extracted layout (top-level folder of each zip in parentheses):
  #   SyRIP.zip (SyRIP/)           -> $RAW/SyRIP/{README.md,annotations,images}
  #   syrip_for_train.zip (syrip/) -> $RAW/syrip_for_train/syrip/{annotations,images}
  # `unzip -o` overwrites any partial earlier extraction in place (no rm: other jobs may be reading).
  mkdir -p "$RAW/syrip_for_train"
  log "unzip SyRIP.zip -> $RAW/SyRIP"
  unzip -q -o "$DL/SyRIP.zip" -d "$RAW"
  log "unzip syrip_for_train.zip -> $RAW/syrip_for_train/syrip"
  unzip -q -o "$DL/syrip_for_train.zip" -d "$RAW/syrip_for_train"
  # sanity: file counts must match the zip central directories
  for z in SyRIP:SyRIP.zip syrip_for_train:syrip_for_train.zip; do
    d=${z%%:*}; zf=${z##*:}
    n_zip=$(unzip -Z1 "$DL/$zf" | grep -v '/$' | wc -l)
    n_disk=$(find "$RAW/$d" -type f | wc -l)
    log "$zf: $n_zip files in zip, $n_disk files on disk"
    [[ "$n_zip" == "$n_disk" ]] || { log "ERROR: count mismatch for $zf"; exit 1; }
  done
  touch "$RAW/UNZIP_DONE"
}

stage_layout() {
  # Upstream layout expected by third_party/Infant-Pose-Estimation (DATASET.ROOT = $SY):
  #   annotations/person_keypoints_<set>.json ; images/<set>/<file>
  # Official sets come straight from the zips; val400/test100/validate500 are written by make_splits.py.
  local S=$RAW/SyRIP T=$RAW/syrip_for_train/syrip
  [[ -f "$RAW/UNZIP_DONE" ]] || { log "ERROR: run the unzip stage first"; exit 1; }
  mkdir -p "$SY/annotations" "$SY/images"
  # train (200 real + 1000 synthetic): official SyRIP annotation, images from SyRIP.zip
  ln -sfn "$S/annotations/200R_1000S/person_keypoints_train_infant.json" "$SY/annotations/person_keypoints_train_infant.json"
  ln -sfn "$S/images/train_infant" "$SY/images/train_infant"
  # domain-classifier pre-training set (adult real vs synthetic): only in syrip_for_train.zip
  ln -sfn "$T/annotations/person_keypoints_train_pre_infant.json" "$SY/annotations/person_keypoints_train_pre_infant.json"
  ln -sfn "$T/images/train_pre_infant" "$SY/images/train_pre_infant"
  # validate_infant = upstream's TEST_SET name = official validate100 = the paper's Test100
  ln -sfn "$S/annotations/validate100/person_keypoints_validate_infant.json" "$SY/annotations/person_keypoints_validate_infant.json"
  ln -sfn "$T/images/validate_infant" "$SY/images/validate_infant"
  log "layout written to $SY"
}

stage_neardups() {
  # optional: regenerate splits/val400_exclude_near_dups.txt (needs cv2: fidip env). The list is committed.
  PYTHONNOUSERSITE=1 "$R/envs/fidip/bin/python" -I "$REPO/scripts/near_dups.py" --raw "$RAW" \
    --out "$SPLITS/near_dups" --thr 10 --oks 0.85 --exclude-out "$REPO/splits/val400_exclude_near_dups.txt"
}

stage_verifyraw() {
  "$PY" -I "$REPO/scripts/verify_raw.py" --downloads "$DL" --raw "$RAW" --sample 60
}

stage_splits() {
  "$PY" -I "$REPO/scripts/make_splits.py" --raw "$RAW" --syrip "$SY" --out "$SPLITS" \
    --ids-out "$REPO/splits" --exclude-val-ids "$REPO/splits/val400_exclude_near_dups.txt"
  touch "$R/data/SPLITS_READY"
  log "touched $R/data/SPLITS_READY"
}

stages=("$@")
[[ ${#stages[@]} -gt 0 ]] || stages=(verify unzip verifyraw layout splits)
for s in "${stages[@]}"; do
  log "=== stage $s ==="
  "stage_$s"
done
log "done: ${stages[*]}"
