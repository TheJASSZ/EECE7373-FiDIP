import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.pose_estimators import TopdownPoseEstimator
from mmpose.registry import MODELS


class _GradReverse(torch.autograd.Function):
    """Identity forward; multiplies the gradient by -lambda backward (Ganin & Lempitsky 2015)."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


@MODELS.register_module()
class DomainAdaptiveTopdown(TopdownPoseEstimator):
    """Top-down heatmap pose estimator with an adversarial domain classifier.

    loss = pose heatmap loss + loss_weight * BCE(D(GRL_lambda(f)), is_synthetic)
    where f is the backbone's last feature map. The classifier D learns to tell real from
    synthetic crops; the reversed gradient pushes the backbone towards domain-invariant features.
    lambda follows the DANN ramp 2/(1+exp(-10p))-1, p = iter / lambda_warmup_iters (clipped to 1).
    Only training changes: predict() is the parent's, and D is unused at test time.

    Args (``da_cfg``): in_channels (backbone channels), hidden (256), loss_weight (0.1),
        grl_lambda (max lambda, 1.0), lambda_warmup_iters (500).
    """

    def __init__(self, *args, da_cfg: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        da_cfg = dict(da_cfg or {})
        c = da_cfg.get('in_channels', 1024)
        hid = da_cfg.get('hidden', 256)
        self.da_loss_weight = float(da_cfg.get('loss_weight', 0.1))
        self.da_max_lambda = float(da_cfg.get('grl_lambda', 1.0))
        self.da_warmup_iters = int(da_cfg.get('lambda_warmup_iters', 500))
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(c, hid, kernel_size=1), nn.BatchNorm2d(hid), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hid, hid), nn.ReLU(inplace=True), nn.Linear(hid, 1))
        self.register_buffer('da_iter', torch.zeros((), dtype=torch.long), persistent=True)

    def grl_lambda(self) -> float:
        p = min(1.0, float(self.da_iter.item()) / max(1, self.da_warmup_iters))
        return self.da_max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

    def loss(self, inputs, data_samples) -> dict:
        feats = self.extract_feat(inputs)
        losses = dict()
        if self.with_head:
            losses.update(self.head.loss(feats, data_samples, train_cfg=self.train_cfg))
        labels = torch.tensor([float(ds.metainfo['is_synthetic']) for ds in data_samples],
                              device=inputs.device)
        lambd = self.grl_lambda()
        self.da_iter += 1
        logits = self.domain_classifier(_GradReverse.apply(feats[-1], lambd)).squeeze(1)
        losses['loss_domain'] = self.da_loss_weight * F.binary_cross_entropy_with_logits(
            logits.float(), labels)
        with torch.no_grad():
            losses['acc_domain'] = ((logits > 0).float() == labels).float().mean()
            losses['grl_lambda'] = torch.tensor(lambd, device=inputs.device)
        return losses
