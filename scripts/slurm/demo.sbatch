#!/bin/bash
# Demo checks (STUDENT_GUIDE Parts 1 and 6): gpu_test.py, tools/demo.py on 8 Test100 images (GT boxes),
# 2 images through the torchvision detector path, a short video, and the Streamlit app verified headless
# (AppTest + a live `streamlit run` health check).
#SBATCH --job-name=rp_demo
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
R=/scratch/purohitham.v/fidip; D=$R/runs/demo; PY=$R/envs/fidip/bin/python
export PYTHONNOUSERSITE=1 TORCH_HOME=$R/models/torch_hub
mkdir -p $D/test100_gt $D/detector $D/in
cd $R/repo/third_party/Infant-Pose-Estimation
echo "== gpu_test.py"; $PY gpu_test.py
# 8 Test100 images, 2 per posture (posture labels from syrip_for_train's copy of validate100)
$PY -I - <<'PYEOF'
import json, os
R = '/scratch/purohitham.v/fidip'
post = {i['id']: i.get('posture') for i in json.load(open(R + '/data/raw/syrip_for_train/syrip/annotations/person_keypoints_validate_infant.json'))['images']}
pick = []
for p in ('supine', 'prone', 'sitting', 'standing'):
    pick += sorted(k for k, v in post.items() if v == p)[:2]
for k in pick:
    dst = R + '/runs/demo/in/test%d.jpg' % k
    if not os.path.lexists(dst):
        os.symlink(R + '/data/syrip/images/test100/test%d.jpg' % k, dst)
print('picked', [(k, post[k]) for k in pick])
PYEOF
CFG=experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml; M=$R/models/hrnet_fidip.pth
echo "== demo.py, GT boxes"; $PY tools/demo.py --cfg $CFG --model $M --image $D/in --output $D/test100_gt --gt-json $R/data/splits/test100.json --conf-threshold 0.3 --output-size 512,512
mkdir -p $D/in_det; for f in $(ls $D/in/*.jpg | head -2); do ln -sfn $(readlink -f $f) $D/in_det/$(basename $f); done
echo "== demo.py, detector path (no GT json)"; $PY tools/demo.py --cfg $CFG --model $M --image $D/in_det --output $D/detector --conf-threshold 0.3 --output-size 512,512
echo "== demo.py, single image"; $PY tools/demo.py --cfg $CFG --model $M --image $D/in/$(ls $D/in | head -1) --output $D/single.jpg
echo "== demo.py, video"; $PY -I -c "
import cv2, glob
fs = sorted(glob.glob('$D/in/*.jpg'))[:4]; w = cv2.VideoWriter('$D/in/clip.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 2, (640, 480))
[w.write(cv2.resize(cv2.imread(f), (640, 480))) for f in fs for _ in range(2)]; w.release()"
$PY tools/demo.py --cfg $CFG --model $M --video $D/in/clip.mp4 --output $D/clip_pose.mp4 --gt-json '' --output-size 640,480
export FIDIP_MODEL=$M FIDIP_CFG=$CFG
echo "== streamlit AppTest"; $PY $R/repo/scripts/test_streamlit_demo.py $D/in/$(ls $D/in | grep jpg | head -1) $R/data/splits/test100.json
echo "== streamlit run + health"; FIDIP_MODEL=$M $R/envs/fidip/bin/streamlit run tools/streamlit_demo.py --server.headless true --server.port 8599 > $D/streamlit_server.log 2>&1 &
SP=$!; for i in $(seq 1 30); do sleep 2; h=$(curl -s --noproxy '*' localhost:8599/_stcore/health || true); [[ $h == ok ]] && break; done
echo "health: $h"; curl -s --noproxy '*' localhost:8599 | grep -o "<title>[^<]*</title>" || true; kill $SP
ls -la $D/test100_gt $D/detector
