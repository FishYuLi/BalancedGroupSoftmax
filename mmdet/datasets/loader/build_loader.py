import platform
from functools import partial

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader

from .sampler import DistributedGroupSampler, DistributedSampler, GroupSampler, GroupSampler_addrepeat, DistributedGroupSampler_addrepeat, DistributedGroupSampler_addrepeat_sampleout

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     **kwargs):
    shuffle = kwargs.get('shuffle', True)
    use_img_sampling = kwargs.get('use_img_sampling', False)
    use_sample_out = kwargs.get('use_sample_out', False)
    if 'use_img_sampling' in kwargs:
        kwargs.pop('use_img_sampling')
    if 'use_sample_out' in kwargs:
        kwargs.pop('use_sample_out')
    if dist:
        rank, world_size = get_dist_info()
        if shuffle:
            if not use_img_sampling:
                print('Dist-train --- Not using image sampling.')
                sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
                                                  world_size, rank)
            else:
                print('Dist-train --- Using image sampling.')
                if use_sample_out:
                    print('Dist-train --- Using sample out.')
                    sampler = DistributedGroupSampler_addrepeat_sampleout(dataset,
                                                            imgs_per_gpu,
                                                            0.001,
                                                            world_size,
                                                            rank)
                else:
                    print('Dist-train --- Not using sample out.')
                    sampler = DistributedGroupSampler_addrepeat(dataset,
                                                            imgs_per_gpu,
                                                            0.001,
                                                            world_size,
                                                            rank)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        if not use_img_sampling:
            print('Not using image sampling.')
            sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None

        else:
            print('Using image sampling.')
            sampler = GroupSampler_addrepeat(dataset, imgs_per_gpu) if shuffle else None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader
