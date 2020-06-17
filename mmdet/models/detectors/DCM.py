from ..registry import DETECTORS
from .two_stage import TwoStageDetector

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi

from mmdet.core import multiclass_nms


@DETECTORS.register_module
class DCM(TwoStageDetector):

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
        super(DCM, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)


        # centers = torch.load('./data/lvis/dcm_center_cls.pt')
        centers = torch.load('./data/lvis/dcm_center_fea.pt')
        centers = centers[1:, :]
        centers = centers / centers.norm(dim=1, keepdim=True)
        self.centers = centers.t() # [fea_dim, 1230]


    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        savename = img_meta[0]['filename'].split('/')[-1].split('.')[0]
        x = self.extract_feat(img)

        losses = dict()

        rois = bbox2roi(gt_bboxes)
        bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        savepath = './DCM/{}-fea.pt'.format(savename)
        torch.save(bbox_feats, savepath)

        cls, bbox, x_cls = self.bbox_head(bbox_feats)


        savepath = './DCM/{}-cls.pt'.format(savename)
        torch.save(x_cls, savepath)


        labels = torch.cat(gt_labels, dim=0)
        savepath = './DCM/{}-lab.pt'.format(savename)
        torch.save(labels, savepath)

        losses['loss'] = cls

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)
        scale_factor = img_meta[0]['scale_factor']

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels, scores = self.simple_test_bboxes(
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        proposals = proposal_list[0] / scale_factor
        pred_label = scores.argmax(dim=1)
        if not self.with_mask:
            return bbox_results, proposals, pred_label
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def simple_test_bboxes0(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred, fea = self.bbox_head(roi_feats)
        cls_score = F.softmax(cls_score, dim=1)
        bg_score = cls_score.narrow(1, 0, 1)

        norm_fea = fea / fea.norm(dim=1, keepdim=True)
        dcm_score = norm_fea.mm(self.centers)

        dcm_score = F.softmax(dcm_score, dim=1)
        dcm_score = (1 - bg_score) * dcm_score

        new_scores = torch.cat([bg_score, dcm_score], dim=1)

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        bboxes, scores = self.bbox_head.get_det_bboxes(
            rois,
            new_scores,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=None)
        # cfg=rcnn_test_cfg)

        det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels, scores


    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)

        cls_score, bbox_pred, fea = self.bbox_head(roi_feats)
        cls_score = F.softmax(cls_score, dim=1)
        bg_score = cls_score.narrow(1, 0, 1)

        num_roi = cls_score.shape[0]
        fea = roi_feats.view(num_roi, -1)
        norm_fea = fea / fea.norm(dim=1, keepdim=True)
        dcm_score = norm_fea.mm(self.centers)

        dcm_score = F.softmax(dcm_score, dim=1)
        dcm_score = (1 - bg_score) * dcm_score

        new_scores = torch.cat([bg_score, dcm_score], dim=1)

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        bboxes, scores = self.bbox_head.get_det_bboxes(
            rois,
            new_scores,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=None)
        # cfg=rcnn_test_cfg)

        det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels, scores