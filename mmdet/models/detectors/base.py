import logging
from abc import ABCMeta, abstractmethod
import os

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn

from mmdet.core import auto_fp16, get_classes, tensor2imgs


class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def show_result(self, data, result, dataset=None, score_thr=0.3):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        class_names = (
        'accordion', 'airplane', 'ant', 'antelope',
        'apple', 'armadillo', 'artichoke', 'axe', 'baby bed', 'backpack',
        'bagel', 'balance beam', 'banana', 'band aid', 'banjo', 'baseball',
        'basketball', 'bathing cap', 'beaker', 'bear', 'bee', 'bell peper',
        'bench', 'bicycle', 'binder', 'bird', 'bookshelf', 'bow tie', 'bow',
        'bowl', 'brassiere', 'burrito', 'bus', 'butterfly', 'camel',
        'can opener', 'car', 'cart', 'cattle', 'cello', 'centipede',
        'chain saw', 'chair', 'chime', 'cocktail shaker', 'coffee maker',
        'computer keyboard', 'computer mouse', 'corkscrew', 'cream',
        'croquet ball', 'crutch', 'cucumber', 'cup or mug', 'diaper',
        'digital clock', 'dishwasher', 'dog', 'domestic cat', 'dragonfly',
        'drum', 'dumbbell', 'electric fan', 'elephant', 'face power',
        'fig', 'filing cabinet', 'flower pot', 'flute', 'fox',
        'french horn', 'frog', 'frying pan', 'giant pada', 'goldfish',
        'golf ball', 'golfcart', 'guacamole', 'guitar', 'hair dryer',
        'hair spray', 'hamburger', 'hammer', 'hamster', 'harmonica',
        'harp', 'hat with a wide brim', 'head cabbage', 'helmet',
        'hippopotamus', 'horizontal bar', 'horse', 'hotdog', 'iPod',
        'isopod', 'jellyfish', 'koala bear', 'ladle', 'ladybug', 'lamp',
        'laptop', 'lemon', 'lion', 'lipstick', 'lizard', 'lobster',
        'maillot', 'maraca', 'microphone', 'microwave', 'milk can',
        'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail',
        'neck brace', 'oboe', 'orange', 'otter', 'pencil box',
        'pencil sharpener', 'perfume', 'person', 'piano', 'pineapple',
        'ping-pong ball', 'pitcher', 'pizza', 'plastic bag', 'plate rack',
        'pomegranate', 'popsicle', 'porcupine', 'power drill', 'pretzel',
        'printer', 'puck', 'punching bag', 'purse', 'rabbit', 'racket',
        'ray', 'red panda', 'refrigerator', 'remote control', 'rubber eraser',
        'rugby ball', 'ruler', 'salt or pepper shaker', 'saxophone',
        'scorpion', 'screwdriver', 'seal', 'sheep', 'ski', 'skunk',
        'snail', 'snake', 'snowmobile', 'snowplow', 'soap dispenser',
        'soccer ball', 'sofa', 'spatula', 'squirrel', 'starfish',
        'stethoscope', 'stove', 'strainer', 'strawberry', 'stretcher',
        'sunglasses', 'swimming trunks', 'swine', 'syringe', 'table',
        'tape palyer', 'tennis ball', 'tick', 'tie', 'tiger', 'toaster',
        'traffic light', 'train', 'trombone', 'trumpet', 'turtle',
        'tv or monitor', 'unicycle', 'vacuum', 'violin', 'volleyball',
        'waffle iron', 'washer', 'water bottle', 'watercraft', 'whale',
        'wine bottle', 'zebra')

        save_dir = './draw/set40/'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # catids = {1, 7, 9, 10, 14, 15, 21, 22, 31, 38, 39, 40, 42, 46, 49, 51, 52, 64, 65, 70, 72, 74, 76, 83, 86, 94, 98, 100, 101, 105, 107, 113, 116, 117, 120, 122, 125, 127, 130, 131, 134, 136, 140, 144, 147, 149, 150, 155, 159, 163, 164, 167, 168, 169, 173, 182, 195, 196, 199, 203, 206, 209, 213, 214, 215, 217, 218, 219, 226, 227, 228, 231, 236, 238, 239, 241, 242, 243, 245, 246, 250, 252, 258, 259, 266, 270, 280, 284, 287, 291, 293, 295, 296, 298, 300, 303, 304, 306, 307, 310, 311, 313, 316, 317, 318, 320, 321, 322, 326, 329, 330, 338, 342, 346, 350, 351, 354, 355, 356, 358, 359, 360, 364, 366, 368, 372, 378, 379, 385, 386, 388, 389, 393, 396, 401, 402, 403, 404, 406, 408, 411, 413, 414, 417, 421, 423, 427, 430, 433, 434, 435, 438, 439, 446, 456, 461, 462, 464, 469, 471, 472, 473, 476, 483, 485, 486, 487, 488, 490, 493, 495, 498, 509, 511, 512, 514, 518, 521, 524, 525, 526, 527, 530, 534, 541, 542, 543, 545, 548, 551, 552, 553, 555, 562, 564, 569, 572, 573, 581, 582, 584, 585, 586, 587, 590, 592, 593, 594, 596, 597, 599, 600, 602, 609, 610, 612, 613, 616, 617, 620, 626, 627, 629, 630, 631, 634, 636, 643, 645, 646, 650, 656, 658, 659, 664, 665, 671, 674, 676, 677, 683, 684, 686, 690, 698, 700, 703, 713, 722, 724, 725, 727, 731, 734, 735, 739, 741, 742, 745, 749, 755, 759, 767, 772, 773, 774, 777, 782, 783, 785, 790, 791, 795, 796, 797, 800, 804, 806, 809, 816, 821, 822, 823, 825, 826, 828, 834, 836, 837, 838, 841, 843, 845, 847, 857, 863, 864, 865, 866, 867, 869, 870, 872, 876, 878, 883, 887, 893, 894, 899, 901, 902, 906, 908, 911, 916, 919, 921, 922, 923, 927, 928, 931, 932, 934, 940, 941, 946, 947, 949, 951, 952, 954, 955, 956, 957, 959, 960, 962, 963, 964, 970, 975, 976, 989, 992, 999, 1000, 1002, 1006, 1009, 1010, 1011, 1013, 1016, 1021, 1023, 1026, 1027, 1029, 1030, 1033, 1034, 1047, 1048, 1049, 1050, 1051, 1056, 1059, 1067, 1068, 1069, 1073, 1074, 1077, 1078, 1082, 1083, 1095, 1100, 1104, 1108, 1136, 1138, 1139, 1140, 1141, 1145, 1147, 1152, 1153, 1154, 1157, 1166, 1167, 1168, 1170, 1172, 1179, 1180, 1181, 1187, 1188, 1190, 1204, 1205, 1212, 1216, 1219, 1226, 1228}
        # import pdb
        # pdb.set_trace()
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']

            img_show = img[:h, :w, :]

            # for i in range(len(bbox_result)):
            #     if (i + 1) not in catids:
            #        bbox_result[i] = np.zeros((0,5))

            bboxes = np.vstack(bbox_result)

            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            savename = img_meta['filename'].split('/')[-1].split('.')[0]
            savename = os.path.join(save_dir, savename+".jpg")
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names= None,#class_names,
                score_thr=0.1,
                show=False,
                out_file=savename,
                bbox_color='green',
                text_color='green',
                thickness=3,
                font_scale=0.8
            )
