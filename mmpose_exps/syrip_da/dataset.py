from typing import Optional

from mmpose.datasets.datasets.body import CocoDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class SyRIPCocoDataset(CocoDataset):
    """COCO-style SyRIP dataset whose instances carry the image-level domain label.

    ``is_synthetic`` (0 = real, 1 = synthetic) is read from ``images[].is_synthetic`` of the GT
    json; add it to ``PackPoseInputs(meta_keys=...)`` to make it visible to the model.
    """

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        info = super().parse_data_info(raw_data_info)
        if info is not None:
            info['is_synthetic'] = int(bool(raw_data_info['raw_img_info'].get('is_synthetic', False)))
        return info
