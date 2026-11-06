# FiDIP Streamlit demo (EECE 7373, STUDENT_GUIDE Part 6): upload an infant image, set the confidence
# threshold, see the skeleton overlay and the per-keypoint confidence table.
#   cd third_party/Infant-Pose-Estimation && streamlit run tools/streamlit_demo.py
# Defaults can be set with FIDIP_CFG / FIDIP_MODEL / FIDIP_GT_JSON. The model is loaded once (cached).
import os
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import KPT_NAMES, PersonDetector, PoseEstimator, draw, gt_boxes  # noqa: E402

st.set_page_config(page_title='FiDIP infant pose demo', layout='wide')
st.title('FiDIP infant pose estimation')

cfg_file = st.sidebar.text_input('Config', os.environ.get('FIDIP_CFG', 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'))
model_file = st.sidebar.text_input('Checkpoint', os.environ.get('FIDIP_MODEL', 'models/hrnet_fidip.pth'))
gt_json = st.sidebar.text_input('GT json (optional, for GT boxes)', os.environ.get('FIDIP_GT_JSON', ''))
use_det = st.sidebar.checkbox('Person detector when no GT box', value=True)
thr = st.sidebar.slider('Keypoint confidence threshold', 0.0, 1.0, 0.3, 0.05)


@st.cache_resource
def load_estimator(cfg_path, model_path):
    import torch
    return PoseEstimator(cfg_path, model_path, 'cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def load_detector():
    import torch
    try:
        return PersonDetector('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning('person detector unavailable: %s' % e)
        return None


up = st.file_uploader('Infant image', type=['jpg', 'jpeg', 'png'])
path = st.text_input('...or an image path on the server', '')
img, name = None, ''
if up is not None:
    img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    name = up.name
elif path:
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    name = os.path.basename(path)
    if img is None:
        st.error('cannot read %s' % path)

if img is not None:
    est = load_estimator(cfg_file, model_file)
    boxes = gt_boxes(gt_json).get(name) if gt_json else None
    src = 'GT box'
    if not boxes:
        det = load_detector() if use_det else None
        b = det(img) if det is not None else None
        boxes, src = ([b], 'detector') if b is not None else ([[0, 0, img.shape[1], img.shape[0]]], 'whole image')
    kpts = est.estimate(img, boxes[0])
    overlay = draw(img, kpts, thr, boxes[0], src)
    c1, c2 = st.columns([3, 2])
    c1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption='%s (box: %s)' % (name, src), use_column_width=True)
    table = pd.DataFrame({'keypoint': KPT_NAMES, 'x': kpts[:, 0].round(1), 'y': kpts[:, 1].round(1),
                          'confidence': kpts[:, 2].round(3), 'drawn': kpts[:, 2] >= thr})
    c2.dataframe(table, hide_index=True, use_container_width=True)
    c2.metric('keypoints above threshold', '%d / 17' % int((kpts[:, 2] >= thr).sum()))
