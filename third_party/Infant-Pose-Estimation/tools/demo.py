# ------------------------------------------------------------------------------
# FiDIP demo (EECE 7373, STUDENT_GUIDE Part 6): image(s) or video + checkpoint -> skeleton overlay.
# Not part of the upstream release; added for the course project. Uses the same preprocessing
# (box -> center/scale with 1.25 padding, affine crop), flip test and DARK post-processing
# (core.inference.get_final_preds) as tools/test_adaptive_model.py, so overlays match the evaluated preds.
#
#   python tools/demo.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
#       --image path/to/img.jpg --model models/hrnet_fidip.pth --output out.jpg \
#       [--conf-threshold 0.3] [--output-size 512,512] [--gt-json coco_gt.json] [--video in.mp4]
#
# Person box: GT box if the image's file name is in --gt-json; else a torchvision Faster R-CNN person
# detector (COCO weights, downloaded to $TORCH_HOME on first use); else the whole image.
# --image may also be a directory (then --output is a directory). A <output>.json with the 17 keypoints
# (x, y, confidence) is written next to each overlay.
# ------------------------------------------------------------------------------
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import cfg, update_config  # noqa: E402
from core.inference import get_final_preds  # noqa: E402
from utils.transforms import flip_back, get_affine_transform  # noqa: E402
import models  # noqa: E402,F401

KPT_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
             'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
             'right_knee', 'left_ankle', 'right_ankle']
SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
            [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]   # 1-based COCO links
FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
LEFT, RIGHT, MID = (255, 128, 0), (0, 128, 255), (0, 200, 0)   # BGR: left limbs blue-ish, right orange-ish


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='FiDIP infant pose demo')
    p.add_argument('--cfg', required=True)
    p.add_argument('--model', required=True, help='pose-net state_dict (e.g. models/hrnet_fidip.pth)')
    p.add_argument('--image', help='image file, directory or glob')
    p.add_argument('--video', help='video file (frames are processed one by one)')
    p.add_argument('--output', required=True, help='output image/video path (directory if several images)')
    p.add_argument('--conf-threshold', type=float, default=0.3, help='draw keypoints with confidence >= this')
    p.add_argument('--output-size', default='', help='W,H: letterbox the overlay to this size (e.g. 512,512)')
    p.add_argument('--gt-json', default='', help='COCO keypoint json: use its box when the file name matches')
    p.add_argument('--detector', choices=['auto', 'none'], default='auto',
                   help='auto: torchvision person detector when no GT box; none: whole image')
    p.add_argument('--det-threshold', type=float, default=0.5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--ext', default='png', help='overlay format when --image is a directory/glob')
    p.add_argument('opts', nargs=argparse.REMAINDER, default=None, help='extra yacs options')
    return p.parse_args(argv)


class PoseEstimator:
    """Loads the FiDIP pose net once; estimate(img_bgr, box_xywh) -> (17,3) array of x, y, confidence."""

    def __init__(self, cfg_file, model_file, device='cuda', opts=None):
        args = argparse.Namespace(cfg=cfg_file, opts=opts or [], modelDir='', logDir='', dataDir='')
        self.cfg = cfg.clone()
        update_config(self.cfg, args)
        self.device = torch.device(device)
        model_p, _ = eval('models.' + self.cfg.MODEL.NAME + '.get_adaptive_pose_net')(self.cfg, is_train=False)
        state = torch.load(model_file, map_location='cpu', weights_only=False)
        if len(state) and all(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        model_p.load_state_dict(state, strict=True)
        self.model = model_p.to(self.device).eval()
        self.w, self.h = int(self.cfg.MODEL.IMAGE_SIZE[0]), int(self.cfg.MODEL.IMAGE_SIZE[1])
        self.tf = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def box_to_cs(self, box):
        x, y, w, h = box
        aspect = self.w * 1.0 / self.h
        center = np.array([x + (w - 1) * 0.5, y + (h - 1) * 0.5], dtype=np.float32)
        if w > aspect * h:
            h = w / aspect
        elif w < aspect * h:
            w = h * aspect
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32) * 1.25   # same as SYRIPDataset._xywh2cs
        return center, scale

    @torch.no_grad()
    def estimate(self, img_bgr, box):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if self.cfg.DATASET.COLOR_RGB else img_bgr
        c, s = self.box_to_cs(box)
        trans = get_affine_transform(c, s, 0, (self.w, self.h))
        crop = cv2.warpAffine(img, trans, (self.w, self.h), flags=cv2.INTER_LINEAR)
        inp = self.tf(crop).unsqueeze(0).to(self.device)
        _, hm = self.model(inp)
        if self.cfg.TEST.FLIP_TEST:
            _, hm_f = self.model(torch.flip(inp, [3]))
            hm_f = torch.from_numpy(flip_back(hm_f.cpu().numpy(), FLIP_PAIRS).copy()).to(self.device)
            hm = (hm + hm_f) * 0.5
        preds, maxvals = get_final_preds(self.cfg, hm.cpu().numpy(), np.array([c]), np.array([s]))
        return np.concatenate([preds[0], maxvals[0]], axis=1)   # (17, 3)


class PersonDetector:
    """torchvision Faster R-CNN (COCO); returns the highest-scoring person box (x, y, w, h) or None."""

    def __init__(self, device, threshold=0.5):
        import torchvision
        w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w).to(device).eval()
        self.device, self.threshold = device, threshold

    @torch.no_grad()
    def __call__(self, img_bgr):
        x = T.functional.to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to(self.device)
        out = self.model([x])[0]
        keep = (out['labels'] == 1) & (out['scores'] >= self.threshold)
        if not keep.any():
            return None
        i = int(torch.argmax(out['scores'] * keep))
        x1, y1, x2, y2 = out['boxes'][i].tolist()
        return [x1, y1, x2 - x1, y2 - y1]


def gt_boxes(gt_json):
    if not gt_json:
        return {}
    d = json.load(open(gt_json))
    by_img = {}
    for a in d['annotations']:
        by_img.setdefault(a['image_id'], []).append(a['bbox'])
    return {os.path.basename(im['file_name']): by_img.get(im['id'], []) for im in d['images']}


def draw(img, kpts, thr, box=None, box_src=''):
    out = img.copy()
    t = max(2, int(round(max(img.shape[:2]) / 300)))
    if box is not None:
        x, y, w, h = [int(round(v)) for v in box]
        cv2.rectangle(out, (x, y), (x + w, y + h), (200, 200, 200), max(1, t // 2))
        if box_src:
            cv2.putText(out, box_src, (x + 3, y + 8 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.25 * t, (200, 200, 200), max(1, t // 2))
    for a, b in SKELETON:
        a, b = a - 1, b - 1
        if kpts[a, 2] >= thr and kpts[b, 2] >= thr:
            col = LEFT if (a % 2 == 1 and b % 2 == 1) else RIGHT if (a % 2 == 0 and b % 2 == 0 and a > 0 and b > 0) else MID
            cv2.line(out, (int(kpts[a, 0]), int(kpts[a, 1])), (int(kpts[b, 0]), int(kpts[b, 1])), col, t, cv2.LINE_AA)
    for j in range(17):
        if kpts[j, 2] >= thr:
            col = MID if j == 0 else LEFT if j % 2 == 1 else RIGHT
            cv2.circle(out, (int(kpts[j, 0]), int(kpts[j, 1])), t + 2, col, -1, cv2.LINE_AA)
            cv2.circle(out, (int(kpts[j, 0]), int(kpts[j, 1])), t + 2, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def letterbox(img, size):
    if not size:
        return img
    W, H = [int(v) for v in size.split(',')]
    r = min(W / img.shape[1], H / img.shape[0])
    im = cv2.resize(img, (max(1, int(img.shape[1] * r)), max(1, int(img.shape[0] * r))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), np.uint8)
    y0, x0 = (H - im.shape[0]) // 2, (W - im.shape[1]) // 2
    canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return canvas


def run_image(est, det, img, name, gts, args):
    if gts.get(name):
        boxes, src = gts[name], 'GT box'
    else:
        b = det(img) if det is not None else None
        boxes, src = ([b], 'detector') if b is not None else ([[0, 0, img.shape[1], img.shape[0]]], 'whole image')
    out, people = img, []
    for box in boxes:
        k = est.estimate(img, box)
        out = draw(out, k, args.conf_threshold, box, src)
        people.append({'box_xywh': [float(v) for v in box], 'box_source': src,
                       'keypoints': {n: [float(k[j, 0]), float(k[j, 1]), float(k[j, 2])] for j, n in enumerate(KPT_NAMES)}})
    return letterbox(out, args.output_size), people


def main(argv=None):
    args = parse_args(argv)
    est = PoseEstimator(args.cfg, args.model, args.device, args.opts)
    gts = gt_boxes(args.gt_json)
    det = None
    if args.detector == 'auto':
        try:
            det = PersonDetector(args.device, args.det_threshold)
        except Exception as e:  # noqa: BLE001  (no internet / no weights: fall back to GT box or whole image)
            print('person detector unavailable (%s); using GT boxes or the whole image' % e)
    if args.video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer, n = None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, _ = run_image(est, det, frame, '', {}, args)
            if writer is None:
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out.shape[1], out.shape[0]))
            writer.write(out)
            n += 1
        if writer is not None:
            writer.release()
        print('wrote %d frames to %s' % (n, args.output))
        return
    if os.path.isdir(args.image):
        files = sorted(f for f in glob.glob(os.path.join(args.image, '*')) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    else:
        files = sorted(glob.glob(args.image)) or [args.image]
    multi = len(files) > 1 or os.path.isdir(args.output)
    if multi:
        os.makedirs(args.output, exist_ok=True)
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            print('cannot read %s' % f)
            continue
        out, people = run_image(est, det, img, os.path.basename(f), gts, args)
        dst = os.path.join(args.output, os.path.splitext(os.path.basename(f))[0] + '_pose.' + args.ext) if multi else args.output
        cv2.imwrite(dst, out)
        json.dump({'image': f, 'conf_threshold': args.conf_threshold, 'people': people},
                  open(os.path.splitext(dst)[0] + '.json', 'w'), indent=1)
        print('%s -> %s (%s, mean conf %.3f)' % (f, dst, people[0]['box_source'],
                                                  np.mean([v[2] for v in people[0]['keypoints'].values()])))


if __name__ == '__main__':
    main()
