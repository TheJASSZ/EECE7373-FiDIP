"""SyRIP domain-adaptation add-ons for MMPose 1.3.2 (IMPROVE stream, stretch goal).

* SyRIPCocoDataset        CocoDataset that carries images[].is_synthetic into every instance.
* DomainAdaptiveTopdown   TopdownPoseEstimator + adversarial real-vs-synthetic domain classifier
                          on the backbone's last feature map through a gradient-reversal layer
                          (DANN). This is the single-loop analogue of FiDIP's alternating
                          domain-classifier / domain-confusion updates. Inference is unchanged.
Load via  custom_imports=dict(imports=['syrip_da'])  with mmpose_exps/ on PYTHONPATH.
"""
from .dataset import SyRIPCocoDataset  # noqa: F401
from .model import DomainAdaptiveTopdown  # noqa: F401
