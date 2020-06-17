import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from mmdet.core import (delta2bbox, force_fp32,
                        multiclass_nms)
from .convfc_bbox_head import SharedFCBBoxHead
from ..builder import build_loss
from ..registry import HEADS
from ..losses import accuracy


@HEADS.register_module
class DCMBBoxHead(SharedFCBBoxHead):

    def __init__(self,
                 num_fcs=2,
                 fc_out_channels=1024,
                 *args,
                 **kwargs):
        super(DCMBBoxHead, self).__init__(num_fcs=num_fcs,
                                               fc_out_channels=fc_out_channels,
                                               *args, **kwargs)


    def forward(self, x):
        # shared part

        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                before_relu = fc(x)
                x = self.relu(before_relu)
        # separate branches
        x_cls = x
        x_reg = x

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, before_relu

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        # if isinstance(cls_score, list):
        #     cls_score = sum(cls_score) / float(len(cls_score))
        # scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        scores = cls_score
        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels