# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Optional

import numpy as np
from mmpose.datasets.datasets.base.base_coco_style_dataset import BaseCocoStyleDataset

from mmpose.registry import DATASETS



@DATASETS.register_module(name="QXCastPoseDatasets")
class QXCastPoseDatasets(BaseCocoStyleDataset):
    """CocoWholeBody dataset for pose estimation.



    COCO-WholeBody keypoints::

        0-10: 11 body keypoints,
        11-53: 42 hand keypoints,

        In total, we have 53 keypoints for wholebody pose estimation.


    """

    METAINFO: dict = dict(
        from_file=r'/workspace/castpose/datasets/qx_castpose.py')