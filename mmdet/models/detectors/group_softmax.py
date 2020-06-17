from ..registry import DETECTORS
from .two_stage import TwoStageDetector

# TODO: delete. This is exactly the same as Faster R-CNN.
#  We can just use the new bbox head with Faster R-CNN or other frameworks.

@DETECTORS.register_module
class GroupSoftmax(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(GroupSoftmax, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
