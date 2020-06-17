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
class ReweightBBoxHead(SharedFCBBoxHead):

    def __init__(self,
                 num_fcs=2,
                 fc_out_channels=1024,
                 reweight_cfg=None,
                 *args,
                 **kwargs):
        super(ReweightBBoxHead, self).__init__(num_fcs=num_fcs,
                                               fc_out_channels=fc_out_channels,
                                               *args, **kwargs)

        self.cls_weight = torch.load(reweight_cfg.cls_weight).cuda()


    def _reweight(self, labels):

        weight = self.cls_weight[labels]

        return weight


    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            new_weight = self._reweight(labels)
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                new_weight,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses