from __future__ import division
import math

import numpy as np
import torch
from mmcv.runner.utils import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler

import pickle
import random

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

class GroupSampler_addrepeat(Sampler):
    ## if combined coco, need to recalculate './class_to_imageid_and_inscount.pt'
    def __init__(self, dataset, samples_per_gpu=1, repeat_t=0.001):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

        self.dataset_class_image_info = pickle.load(open('./data/lvis/class_to_imageid_and_inscount.pt', 'rb'))
        # self.dataset_abundant_class_image_info = [self.dataset_class_image_info[cls_idx]
        #                                           for cls_idx in range(len(self.dataset_class_image_info))
        #                                           if self.dataset_class_image_info[cls_idx]['isntance_count'] > 1000]
        self.dataset_abundant_class_image_info = [self.dataset_class_image_info[cls_idx]
                                                  for cls_idx in range(len(self.dataset_class_image_info))
                                                  if self.dataset_class_image_info[cls_idx]['isntance_count'] > 0]
        self.dataset_abundant_class_ids = [item['category_id'] for item in self.dataset_abundant_class_image_info]

        ## calculate repeating num. per image
        repeat_per_img = {}
        total_img_num = len(self.dataset.img_infos)
        clses_to_repeat = []
        for cls in range(1,1231):
            fc = len(self.dataset_class_image_info[cls]['image_info_id'])/float(total_img_num)
            repeat_this_cls = max(1., np.sqrt(repeat_t/fc))
            if repeat_this_cls>1:
                clses_to_repeat.append(cls)
            for img_info_id in self.dataset_class_image_info[cls]['image_info_id']:
                if img_info_id not in repeat_per_img:
                    repeat_per_img[img_info_id]=repeat_this_cls
                else:
                    if repeat_per_img[img_info_id]<repeat_this_cls:
                        repeat_per_img[img_info_id]=repeat_this_cls
                    else:
                        pass
        repeat_per_img =  {k:math.ceil(v) for i, (k,v) in enumerate(repeat_per_img.items())}## ceiling
        assert len(repeat_per_img.keys()) == total_img_num
        img_info_ids_to_repeat = {k:v for i, (k,v) in enumerate(repeat_per_img.items()) if v>1}## repeat larget than 1 imgs
        self.img_info_ids_to_repeat = img_info_ids_to_repeat

        ## calculate new group size infomation
        self.group_sizes_new=self.group_sizes.copy()
        for i, size in enumerate(self.group_sizes):
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size

            for idx, (img_info_id, re_count) in enumerate(self.img_info_ids_to_repeat.items()):
                if img_info_id in indice:
                    self.group_sizes_new[i]+=re_count

        self.num_samples_new = 0
        for i, size in enumerate(self.group_sizes_new):
            self.num_samples_new += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu



    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
## add repeat imgs
            for idx, (img_info_id, re_count) in enumerate(self.img_info_ids_to_repeat.items()):
                if img_info_id in indice:
                    indice = np.concatenate([indice, [img_info_id]*re_count])

            np.random.shuffle(indice)
            num_extra = int(np.ceil(len(indice) / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)

##
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples_new
        return iter(indices)

    def __len__(self):
        return self.num_samples_new


class EpisodicSampler(Sampler):

    def __init__(self, dataset, batch_size_total, nc, episode):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        # self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        # for i, size in enumerate(self.group_sizes):
        #     self.num_samples += int(np.ceil(
        #         size / self.samples_per_gpu)) * self.samples_per_gpu
        self.dataset_class_image_info = pickle.load(open('./data/lvis/class_to_imageid_and_inscount.pt', 'rb'))
        # self.dataset_abundant_class_image_info = [self.dataset_class_image_info[cls_idx]
        #                                           for cls_idx in range(len(self.dataset_class_image_info))
        #                                           if self.dataset_class_image_info[cls_idx]['isntance_count'] > 1000]
        self.dataset_abundant_class_image_info = [self.dataset_class_image_info[cls_idx]
                                                  for cls_idx in range(len(self.dataset_class_image_info))
                                                  if self.dataset_class_image_info[cls_idx]['isntance_count'] > 0]
        self.dataset_abundant_class_ids = [item['category_id'] for item in self.dataset_abundant_class_image_info]
        self.nc = nc
        self.bs = batch_size_total
        self.episode = episode
    def __iter__(self):
        # indices = []
        # for i, size in enumerate(self.group_sizes):
        #     if size == 0:
        #         continue
        #     indice = np.where(self.flag == i)[0]
        #     assert len(indice) == size
        #     np.random.shuffle(indice)
        #     num_extra = int(np.ceil(size / self.samples_per_gpu)
        #                     ) * self.samples_per_gpu - len(indice)
        #     indice = np.concatenate([indice, indice[:num_extra]])
        #     indices.append(indice)
        # indices = np.concatenate(indices)
        # indices = [
        #     indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
        #     for i in np.random.permutation(
        #         range(len(indices) // self.samples_per_gpu))
        # ]
        # indices = np.concatenate(indices)
        # indices = indices.astype(np.int64).tolist()
        # assert len(indices) == self.num_samples
        # return iter(indices)

        ##first, sample per episode classes(NC) self.bs/self.nc is the num of img per class
        indices = []
        class_indices = []
        for i in range(self.episode):## possiblly need to ensure per gpu images are with similar ratio
            per_episode_cls_sampled = random.sample(self.dataset_abundant_class_image_info, self.nc)
            # per_episode_image_sampled = [random.sample(item['image_info_id'], int(self.bs/self.nc)) for item in per_episode_cls_sampled]
            per_episode_image_sampled = [random.choices(item['image_info_id'], k=int(self.bs / self.nc)) for item in
                                         per_episode_cls_sampled]
            indices.append(np.concatenate(per_episode_image_sampled))
            class_indices.append(np.stack([item['category_id'] for item in per_episode_cls_sampled for i in range(int(self.bs/self.nc))]))
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        self.class_indices = np.concatenate(class_indices)## for external access of current nc classes, this is an ugly way out of dataloader
        assert len(indices) == self.episode*self.bs
        return iter(indices)

    def __len__(self):
        return self.episode*self.bs

class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedGroupSampler_addrepeat(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 repeat_t=0.001,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        # _rank, _num_replicas = 0, 8
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.dataset_class_image_info = pickle.load(open('./data/lvis/class_to_imageid_and_inscount.pt', 'rb'))
        img_id_to_new_img_info_id = {info['id']: idx for idx, info in enumerate(self.dataset.img_infos)}
        for idx, (k,v) in enumerate(self.dataset_class_image_info.items()):
            v_new = v.copy()
            v_new['image_info_id'] = [img_id_to_new_img_info_id[id] for id in v_new['img_id']]
            self.dataset_class_image_info[k] =v_new
        # self.dataset_abundant_class_image_info = [self.dataset_class_image_info[cls_idx]
        #                                           for cls_idx in range(len(self.dataset_class_image_info))
        #                                           if self.dataset_class_image_info[cls_idx]['isntance_count'] > 1000]
        self.dataset_abundant_class_image_info = [self.dataset_class_image_info[cls_idx]
                                                  for cls_idx in range(len(self.dataset_class_image_info))
                                                  if self.dataset_class_image_info[cls_idx]['isntance_count'] > 0]
        self.dataset_abundant_class_ids = [item['category_id'] for item in self.dataset_abundant_class_image_info]

        ## calculate repeating num. per image
        repeat_per_img = {}
        total_img_num = len(self.dataset.img_infos)
        clses_to_repeat = []
        for cls in range(1,1231):
            fc = len(self.dataset_class_image_info[cls]['image_info_id'])/float(total_img_num)
            repeat_this_cls = max(1., np.sqrt(repeat_t/fc))
            if repeat_this_cls>1:
                clses_to_repeat.append(cls)
            for img_info_id in self.dataset_class_image_info[cls]['image_info_id']:
                if img_info_id not in repeat_per_img:
                    repeat_per_img[img_info_id]=repeat_this_cls
                else:
                    if repeat_per_img[img_info_id]<repeat_this_cls:
                        repeat_per_img[img_info_id]=repeat_this_cls
                    else:
                        pass
        repeat_per_img =  {k:math.ceil(v) for i, (k,v) in enumerate(repeat_per_img.items())}## ceiling
        assert len(repeat_per_img.keys()) == total_img_num
        img_info_ids_to_repeat = {k:v for i, (k,v) in enumerate(repeat_per_img.items()) if v>1}## repeat larget than 1 imgs
        self.img_info_ids_to_repeat = img_info_ids_to_repeat

        ## calculate new group size infomation
        self.group_sizes_new=self.group_sizes.copy()
        for i, size in enumerate(self.group_sizes):
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size

            for idx, (img_info_id, re_count) in enumerate(self.img_info_ids_to_repeat.items()):
                if img_info_id in indice:
                    self.group_sizes_new[i]+=re_count

        self.num_samples_new = 0
        for i, j in enumerate(self.group_sizes_new):
            self.num_samples_new += int(
                math.ceil(self.group_sizes_new[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples_new * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                for idx, (img_info_id, re_count) in enumerate(self.img_info_ids_to_repeat.items()):
                    if img_info_id in indice:
                        indice = np.concatenate([indice, [img_info_id] * re_count])

                indice = indice[list(torch.randperm(len(indice),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        len(indice) * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples_new * self.rank
        indices = indices[offset:offset + self.num_samples_new]
        # indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples_new

        return iter(indices)

    def __len__(self):
        return self.num_samples_new

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedGroupSampler_addrepeat_sampleout(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 repeat_t=0.001,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        # _rank, _num_replicas = 0, 8
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0


        self.num_to_sample_out= [6000, 17000]
        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.dataset_class_image_info = pickle.load(open('./data/lvis/class_to_imageid_and_inscount.pt', 'rb'))
        img_id_to_new_img_info_id = {info['id']: idx for idx, info in enumerate(self.dataset.img_infos)}
        for idx, (k,v) in enumerate(self.dataset_class_image_info.items()):
            v_new = v.copy()
            v_new['image_info_id'] = [img_id_to_new_img_info_id[id] for id in v_new['img_id']]
            self.dataset_class_image_info[k] =v_new
        # self.dataset_abundant_class_image_info = [self.dataset_class_image_info[cls_idx]
        #                                           for cls_idx in range(len(self.dataset_class_image_info))
        #                                           if self.dataset_class_image_info[cls_idx]['isntance_count'] > 1000]
        self.dataset_abundant_class_image_info = [self.dataset_class_image_info[cls_idx]
                                                  for cls_idx in range(len(self.dataset_class_image_info))
                                                  if self.dataset_class_image_info[cls_idx]['isntance_count'] > 0]
        self.dataset_abundant_class_ids = [item['category_id'] for item in self.dataset_abundant_class_image_info]

        ## calculate repeating num. per image
        repeat_per_img = {}
        total_img_num = len(self.dataset.img_infos)
        clses_to_repeat = []
        for cls in range(1,1231):
            fc = len(self.dataset_class_image_info[cls]['image_info_id'])/float(total_img_num)
            repeat_this_cls = max(1., np.sqrt(repeat_t/fc))
            if repeat_this_cls>1:
                clses_to_repeat.append(cls)
            for img_info_id in self.dataset_class_image_info[cls]['image_info_id']:
                if img_info_id not in repeat_per_img:
                    repeat_per_img[img_info_id]=repeat_this_cls
                else:
                    if repeat_per_img[img_info_id]<repeat_this_cls:
                        repeat_per_img[img_info_id]=repeat_this_cls
                    else:
                        pass
        repeat_per_img =  {k:math.ceil(v) for i, (k,v) in enumerate(repeat_per_img.items())}## ceiling
        assert len(repeat_per_img.keys()) == total_img_num
        img_info_ids_to_repeat = {k:v for i, (k,v) in enumerate(repeat_per_img.items()) if v>1}## repeat larget than 1 imgs
        self.img_info_ids_to_repeat = img_info_ids_to_repeat

        ## calculate new group size infomation
        self.group_sizes_new=self.group_sizes.copy()
        for i, size in enumerate(self.group_sizes):
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size

            for idx, (img_info_id, re_count) in enumerate(self.img_info_ids_to_repeat.items()):
                if img_info_id in indice:
                    self.group_sizes_new[i]+=re_count

        self.num_samples_new = 0
        for i, j in enumerate(self.group_sizes_new):
            self.num_samples_new += int(
                math.ceil(self.group_sizes_new[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples_new * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice_orig = indice.copy()

                img_info_ids_to_sample = [img_info_id for img_info_id in indice.tolist() if
                                            img_info_id not in list(self.img_info_ids_to_repeat.keys())]
                img_info_ids_sampled = [img_info_ids_to_sample[id] for id in
                list(torch.randperm(len(img_info_ids_to_sample), generator=g))[:(len(img_info_ids_to_sample) - self.num_to_sample_out[i])]]

                # for idx, (img_info_id, re_count) in enumerate(self.img_info_ids_to_repeat.items()):
                #     if img_info_id in indice:
                #         indice = np.concatenate([indice, [img_info_id] * re_count])
                indice = np.array(img_info_ids_sampled)
                for idx, (img_info_id, re_count) in enumerate(self.img_info_ids_to_repeat.items()):
                    if img_info_id in indice_orig:
                        indice = np.concatenate([indice, [img_info_id] * (re_count+1)])## +1 to add once the original img ids of repeated ones

                indice = indice[list(torch.randperm(len(indice),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        len(indice) * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        self.total_size= len(indices)
        self.num_samples_new = int(len(indices)/8)
        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample

        ## sanity check
        # print(indices)
        # cat_ins_num = {}
        # for rank in range(8):
        #     offset = self.num_samples_new * rank
        #     indices_rank = indices[offset:offset + self.num_samples_new]
        #     for img_info_id in indices_rank:
        #         for ann in self.dataset.lvis.img_ann_map[self.dataset.img_infos[img_info_id]['id']]:
        #             if ann['category_id'] not in cat_ins_num:
        #                 cat_ins_num[ann['category_id']] = 0
        #             cat_ins_num[ann['category_id']] += 1

        offset = self.num_samples_new * self.rank
        indices = indices[offset:offset + self.num_samples_new]
        ## print this gpu cat ins num
        cat_ins_num = {}
        for img_info_id in indices:
            for ann in self.dataset.lvis.img_ann_map[self.dataset.img_infos[img_info_id]['id']]:
                if ann['category_id'] not in cat_ins_num:
                    cat_ins_num[ann['category_id']] = 0
                cat_ins_num[ann['category_id']] += 1
        # pickle.dump(cat_ins_num, open('./this_gpu_cat_ins_num_gpu{}.pt'.format(self.rank), 'wb'))
        # print('this gpu cat ins num {}'.format(cat_ins_num))

        # indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples_new


        return iter(indices)

    def __len__(self):
        return self.num_samples_new

    def set_epoch(self, epoch):
        self.epoch = epoch
