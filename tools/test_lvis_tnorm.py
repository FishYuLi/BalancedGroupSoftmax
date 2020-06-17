import argparse
import os
import os.path as osp
import shutil
import tempfile
import pdb
import numpy as np
import pickle

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import lvis_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from mmdet.core import build_assigner


def accumulate_acc(num_ins, num_get, splitbin):
    savelist = [num_get, num_ins]
    # with open('tempcls.pkl', 'wb') as fout:
    #     pickle.dump(savelist, fout)
    print('Saving pro cls result to: {}'.format('tempcls.pkl'))

    print('\n')
    print('========================================================')
    title_format = "| {} | {} | {} | {} | {} | {} |"
    print(title_format.format('Type', 'IoU', 'Area', 'MaxDets', 'CatIds',
                              'Result'))
    print(title_format.format(':---:', ':---:', ':---:', ':---:', ':---:',
                              ':---:'))
    template = "| {:^6} | {:<9} | {:>6s} | {:>3d} | {:>12s} | {:2.2f}% |"
    for k, v in splitbin.items():
        ins_count = num_ins[v].sum().astype(np.float64)
        get_count = num_get[v].sum().astype(np.float64)
        acc = get_count / ins_count
        print(template.format('(ACC)', '0.50:0.95', 'all', 300, k, acc * 100))

def get_split_bin(dataset, num_ins):
    split_file_name = './data/lvis/valsplit.pkl'

    if osp.exists(split_file_name):
        with open(split_file_name, 'rb') as fin:
            splits = pickle.load(fin)
        print('Load split file from: {}'.format(split_file_name))
        return splits

    print('Calculate split file...')
    catsinfo = dataset.lvis.cats

    bin10 = []
    bin100 = []
    bin1000 = []
    binover = []

    for cid, cate in catsinfo.items():
        ins_count = cate['instance_count']
        if num_ins[cid] == 0:
            continue
        if ins_count < 10:
            bin10.append(cid)
        elif ins_count < 100:
            bin100.append(cid)
        elif ins_count < 1000:
            bin1000.append(cid)
        else:
            binover.append(cid)

    splits = {}
    splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['[10, 100)'] = np.array(bin100, dtype=np.int)
    splits['[100, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[1000, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)
    print('Dump split file to: {}'.format(split_file_name))
    return splits


def single_gpu_test(model, data_loader, show=False, cfg=None):
    model.eval()
    results = []
    dataset = data_loader.dataset

    prog_bar = mmcv.ProgressBar(len(dataset))
    box_assigner = build_assigner(cfg.train_cfg.rcnn.assigner)

    num_cls = len(dataset.cat_ids) + 1
    num_ins = np.zeros((num_cls,))
    num_get = np.zeros((num_cls,))

    for i, data in enumerate(data_loader):

        gt = dataset.get_ann_info(i)
        gt_bboxes = gt['bboxes']
        gt_labels = gt['labels']
        gt_bboxes_ignore = gt['bboxes_ignore']
        with torch.no_grad():
            result, proposals, pred_label = model(return_loss=False,
                                                  rescale=not show, **data)
        results.append(result)

        assign_label = np.zeros((proposals.shape[0]))
        pred_label = pred_label.cpu().numpy()
        if gt_bboxes.shape[0] > 0:
            assign_result = box_assigner.assign(proposals,
                                                torch.FloatTensor(
                                                    gt_bboxes).cuda(),
                                                torch.FloatTensor(
                                                    gt_bboxes_ignore).cuda(),
                                                None)
            assign_gt_inds = assign_result.gt_inds.cpu().numpy()  # tensor([1000])
            gt_labels = np.hstack((np.zeros((1,)), gt_labels))
            assign_label = gt_labels[assign_gt_inds]

        matches = (pred_label == assign_label).astype(np.int)
        for ibox in range(pred_label.shape[0]):
            gt_cls = assign_label[ibox].astype(np.int)
            num_ins[gt_cls] += 1
            num_get[gt_cls] += matches[ibox]

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

        # if i > 20:
        #     break

    splitbin = get_split_bin(dataset, num_ins)
    accumulate_acc(num_ins, num_get, splitbin)
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tau', type=float, default=1.0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def reweight_cls(model):
    model_dict = model.state_dict()

    cls_weight = model_dict['bbox_head.fc_cls.weight']  # ([1231, 1024])

    def pnorm(weights, tau):
        normB = torch.norm(weights, 2, 1)
        ws = weights.clone()

        for i in range(1, weights.shape[0]):
            ws[i] = ws[i] / torch.pow(normB[i], tau)

        return ws

    cls_weight = pnorm(cls_weight, 1)
    model_dict['bbox_head.fc_cls.weight'].copy_(cls_weight)
    # pdb.set_trace()

    return model


def reweight_cls_newhead(model, tauuu):
    model_dict = model.state_dict()

    # Copy fcs
    for k, v in model_dict.items():
        if k.startswith('bbox_head.'):
            newname = k.split('.')
            newname[0] = newname[0] + '_back'
            newname = '.'.join(newname)

            model_dict[newname].copy_(v)

            print('Copy param {:<30} to {:<30}'.format(k, newname))

    def pnorm(weights, tau):
        normB = torch.norm(weights, 2, 1)
        ws = weights.clone()

        for i in range(1, weights.shape[0]):
            ws[i] = ws[i] / torch.pow(normB[i], tau)

        return ws

    reweight_set = ['bbox_head_back.fc_cls.weight']
    tau = tauuu
    for k in reweight_set:
        weight = model_dict[k]  # ([1231, 1024])
        weight = pnorm(weight, tau)
        model_dict[k].copy_(weight)
        print('Reweight param {:<30} with tau={}'.format(k, tau))

    return model


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.test_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = reweight_cls_newhead(model, args.tau)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, cfg)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                lvis_eval(result_file, eval_types, dataset.lvis)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs, args.out)
                    lvis_eval(result_files, eval_types, dataset.lvis)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        lvis_eval(result_files, eval_types, dataset.lvis)

    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            results2json(dataset, outputs, args.json_out)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)


if __name__ == '__main__':
    main()
