from lvis.lvis import LVIS
import numpy as np
import pickle
import pdb
import os
import json
import torch
from pycocotools.coco import COCO


def get_cate_gs():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1, 1, 1]
    label2binlabel = np.zeros((5, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 10:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        elif ins_count < 100:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1
        elif ins_count < 1000:
            label2binlabel[3, cid] = binlabel_count[3]
            binlabel_count[3] += 1
        else:
            label2binlabel[4, cid] = binlabel_count[4]
            binlabel_count[4] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((5, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    # pdb.set_trace()

    return pred_slice

def get_split():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin10 = []
    bin100 = []
    bin1000 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
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

    split_file_name = './data/lvis/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)


def ana_param():

    cate2insnum_file = './data/lvis/cate2insnum.pkl'
    if False: # os.path.exists(cate2insnum_file):
        with open(cate2insnum_file, 'rb') as f:
            cate2insnum = pickle.load(f)

    else:
        train_ann_file = './data/lvis/lvis_v0.5_train.json'
        val_ann_file = './data/lvis/lvis_v0.5_val.json'

        lvis_train = LVIS(train_ann_file)
        lvis_val = LVIS(val_ann_file)
        train_catsinfo = lvis_train.cats
        val_catsinfo = lvis_val.cats

        train_cat2ins = [v['instance_count'] for k, v in train_catsinfo.items()]
        train_cat2ins = [0] + train_cat2ins
        val_cat2ins = [v['instance_count'] for k, v in val_catsinfo.items()]
        val_cat2ins = [0] + val_cat2ins

        cate2insnum = {}
        cate2insnum['train'] = np.array(train_cat2ins, dtype=np.int)
        cate2insnum['val'] = np.array(val_cat2ins, dtype=np.int)

        with open(cate2insnum_file, 'wb') as fout:
            pickle.dump(cate2insnum, fout)

    checkpoint_file = './work_dirs/faster_rcnn_r50_fpn_1x_lvis/latest.pth'
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    param = checkpoint['state_dict']
    cls_fc_weight = param['bbox_head.fc_cls.weight'].numpy()
    cls_fc_bias = param['bbox_head.fc_cls.bias'].numpy()

    cls_fc_weight_norm = np.linalg.norm(cls_fc_weight, axis=1)

    savelist = [cls_fc_weight_norm,
                cls_fc_bias]

    with open('./data/lvis/r50_param_ana.pkl', 'wb') as fout:
        pickle.dump(savelist, fout)


def ana_coco_param():

    train_ann_file = './data/coco/annotations/instances_train2017.json'
    val_ann_file = './data/coco/annotations/instances_val2017.json'

    coco_train = COCO(train_ann_file)
    coco_val = COCO(val_ann_file)

    cat2insnum_train = np.zeros((91,), dtype=np.int)
    for k, v in coco_train.imgToAnns.items():
        for term in v:
            cat2insnum_train[term['category_id']] += 1

    cat2insnum_val = np.zeros((91,), dtype=np.int)
    for k, v in coco_val.imgToAnns.items():
        for term in v:
            cat2insnum_val[term['category_id']] += 1

    cat2insnum_train = cat2insnum_train[np.where(cat2insnum_train > 0)[0]]
    cat2insnum_val = cat2insnum_val[np.where(cat2insnum_val > 0)[0]]
    cat2insnum_train = np.hstack((np.zeros((1,), dtype=np.int), cat2insnum_train))
    cat2insnum_val = np.hstack((np.zeros((1,), dtype=np.int), cat2insnum_val))

    checkpoint_file = './data/download_models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    param = checkpoint['state_dict']
    cls_fc_weight = param['bbox_head.fc_cls.weight'].numpy()
    cls_fc_bias = param['bbox_head.fc_cls.bias'].numpy()

    cls_fc_weight_norm = np.linalg.norm(cls_fc_weight, axis=1)

    savedict = {'train_ins': cat2insnum_train,
                'val_ins': cat2insnum_val,
                'weight': cls_fc_weight_norm,
                'bias': cls_fc_bias}

    with open('./localdata/cocoparam.pkl', 'wb') as fout:
        pickle.dump(savedict, fout)

def load_checkpoint():

    # checkpoint_file = './work_dirs/faster_rcnn_r50_fpn_1x_lvis/latest.pth'
    # checkpoint_file = 'data/download_models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'
    checkpoint_file = './work_dirs/faster_rcnn_r50_fpn_1x_lvis_is/epoch_12.pth'
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    param = checkpoint['state_dict']

    cls_fc_weight = param['bbox_head.fc_cls.weight'].numpy()
    cls_fc_bias = param['bbox_head.fc_cls.bias'].numpy()
    cls_fc_weight_norm = np.linalg.norm(cls_fc_weight, axis=1)

    reg_weight = param['bbox_head.fc_reg.weight'].numpy()
    reg_bias = param['bbox_head.fc_reg.bias'].numpy()
    reg_weight_norm = np.linalg.norm(reg_weight, axis=1)
    reg_weight_norm = reg_weight_norm.reshape((81, 4)).mean(axis=1)
    reg_bias = reg_bias.reshape((81, 4)).mean(axis=1)


    savedict = {'cls_weight': cls_fc_weight_norm,
                'cls_bias': cls_fc_bias,
                'reg_weight': reg_weight_norm,
                'reg_bias': reg_bias}

    with open('./localdata/r50_weight_coco.pkl', 'wb') as fout:
        pickle.dump(savedict, fout)


def load_checkpoint_all():

    # checkpoint_file = './work_dirs/faster_rcnn_r50_fpn_1x_lvis/latest.pth'
    # checkpoint_file = 'data/download_models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'
    checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_lvis_reweighthead/epoch_12.pth'
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    param = checkpoint['state_dict']

    cls_fc_weight = param['bbox_head.fc_cls.weight'].numpy()
    cls_fc_weight_norm = np.linalg.norm(cls_fc_weight, axis=1)

    savedict = {'reweight': cls_fc_weight_norm}

    checkpoint_file = './work_dirs/gs_faster_rcnn_r50_fpn_1x_lvis_with0_bg8/epoch_12.pth'
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    param = checkpoint['state_dict']

    cls_fc_weight = param['bbox_head.fc_cls.weight'].numpy()
    new_cls_fc_weight_norm = np.linalg.norm(cls_fc_weight, axis=1) # (1236,)

    # pdb.set_trace()
    pred_slice_file = './data/lvis/pred_slice_with0.pt'
    fg_split = './data/lvis/valsplit.pkl'
    pred_slice = torch.load(pred_slice_file).numpy()
    with open(fg_split, 'rb') as fin:
        fg_split = pickle.load(fin)
    fg_splits = []
    fg_splits.append(fg_split['(0, 10)'])
    fg_splits.append(fg_split['[10, 100)'])
    fg_splits.append(fg_split['[100, 1000)'])
    fg_splits.append(fg_split['[1000, ~)'])

    new_preds = []
    num_bins = pred_slice.shape[0]
    for i in range(num_bins):
        start = pred_slice[i, 0]
        length = pred_slice[i, 1]
        sliced_pred = new_cls_fc_weight_norm[start:start+length]
        new_preds.append(sliced_pred)

    bg = new_preds[0]
    rest = new_preds[1:]
    fg_merge = np.zeros((1231,))
    fg_merge[0] = bg[0]

    for i, split in enumerate(fg_splits):
        fg_merge[split] = rest[i][1:]

    # pdb.set_trace()

    savedict['gs'] = fg_merge
    with open('./localdata/weneedparame.pkl', 'wb') as fout:
        pickle.dump(savedict, fout)

def get_mask():
    train_ann_file = './data/lvis/lvis_v0.5_train.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    mask = np.zeros((1231, ), dtype=np.int)

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 100:
            mask[cid] = 1

    mask_torch = torch.from_numpy(mask)
    torch.save(mask_torch, './data/lvis/mask.pt')


def trymapping():

    mask = torch.load('./data/lvis/mask.pt')

    ids = np.array([0,0,2,0,1], dtype=np.int)
    cls_ids = torch.from_numpy(ids)

    for i in range(5):
        new_ids = mask[cls_ids]
        print(new_ids)


def test_node_map():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    bin10 = []
    bin100 = []
    bin1000 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
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

    split_file_name = './data/lvis/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)


def get_cate_weight():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    ins_count_all = np.zeros((1231,), dtype=np.float)

    for cid, cate in train_catsinfo.items():
        ins_count_all[cid] = cate['instance_count']

    ins_count_all[0] = 1
    tmp = np.ones_like(ins_count_all)

    weight = tmp / ins_count_all
    weight_mean = weight[1:].mean()
    weight = weight / weight_mean
    weight[0] = 1

    # pdb.set_trace()

    weight = np.where(weight > 5, 5, weight)
    weight = np.where(weight < 0.1, 0.1, weight)

    savebin = torch.from_numpy(weight)

    save_path = './data/lvis/cls_weight.pt'
    torch.save(savebin, save_path)

    # pdb.set_trace()


def get_cate_weight_bf():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    beta = 0.999

    ins_count_all = np.zeros((1231,), dtype=np.float)

    for cid, cate in train_catsinfo.items():
        ins_count_all[cid] = cate['instance_count']

    ins_count_all[0] = np.sum(ins_count_all[1:]) * 3
    ins_count = ins_count_all # ins_count_all[1:]
    tmp = np.ones_like(ins_count)

    weight = (tmp - beta) / (tmp - np.power(beta, ins_count))
    weight_mean = np.mean(weight)
    weight = weight / weight_mean
    # weight_mean = weight[1:].mean()
    # weight = weight / weight_mean
    # weight[0] = 1

    # weight_all = np.ones((1231, ), dtype=np.float)
    # weight_all[1:] = weight

    pdb.set_trace()

    # weight = np.where(weight > 5, 5, weight)
    # weight = np.where(weight < 0.1, 0.1, weight)

    savebin = torch.from_numpy(weight)

    save_path = './data/lvis/cls_weight_bf.pt'
    torch.save(savebin, save_path)

    # pdb.set_trace()

def get_cate_weight_bours():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    beta = 0.999

    ins_count_all = np.zeros((1231,), dtype=np.float)

    for cid, cate in train_catsinfo.items():
        ins_count_all[cid] = cate['instance_count']

    ins_count_all[0] = 1
    ins_count = ins_count_all[1:]
    tmp = np.ones_like(ins_count)

    weight = (tmp - beta) / (tmp - np.power(beta, ins_count))
    weight_mean = np.mean(weight)
    weight = weight / weight_mean
    # weight_mean = weight[1:].mean()
    # weight = weight / weight_mean
    # weight[0] = 1

    weight_all = np.ones((1231, ), dtype=np.float)
    weight_all[1:] = weight
    weight = weight_all

    pdb.set_trace()

    weight = np.where(weight > 5, 5, weight)
    weight = np.where(weight < 0.1, 0.1, weight)

    savebin = torch.from_numpy(weight)

    save_path = './data/lvis/cls_weight_bours.pt'
    torch.save(savebin, save_path)

    # pdb.set_trace()

def get_bin_weight():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    ins_count_all = np.zeros((1231,), dtype=np.float)

    for cid, cate in train_catsinfo.items():
        ins_count_all[cid] = cate['instance_count']

    ins_count_all[0] = 1
    tmp = np.ones_like(ins_count_all)

    weight = tmp / ins_count_all

    label2binlabel = torch.load('./data/lvis/label2binlabel.pt').cpu().numpy()
    pdb.set_trace()

    bins = label2binlabel[1:, :]

    allws = []
    for i in range(1, label2binlabel.shape[0]):
        bin = label2binlabel[i]
        idx = np.where(bin > 0)
        binw = weight[idx]
        binw_mean = binw.mean()
        binw = binw / binw_mean
        binw = np.where(binw > 5, 5, binw)
        binw = np.where(binw < 0.1, 0.1, binw)
        binw = np.hstack((np.ones(1,), binw))

        allws.append(binw)

    with open('./data/lvis/bins_cls_weight.pkl', 'wb') as fout:
        pickle.dump(allws, fout)


def get_cate_gs2():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1]
    label2binlabel = np.zeros((3, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 100:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        else:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/2bins/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((3, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/2bins/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    pdb.set_trace()

def get_split2():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin100 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']

        if ins_count < 100:
            bin100.append(cid)
        else:
            binover.append(cid)

    splits = {}
    # splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['[10, 100)'] = np.array(bin100, dtype=np.int)
    # splits['[100, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[1000, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/2bins/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)


def get_cate_gs8():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    label2binlabel = np.zeros((9, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 5:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        elif ins_count < 10:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1
        elif ins_count < 50:
            label2binlabel[3, cid] = binlabel_count[3]
            binlabel_count[3] += 1
        elif ins_count < 100:
            label2binlabel[4, cid] = binlabel_count[4]
            binlabel_count[4] += 1
        elif ins_count < 500:
            label2binlabel[5, cid] = binlabel_count[5]
            binlabel_count[5] += 1
        elif ins_count < 1000:
            label2binlabel[6, cid] = binlabel_count[6]
            binlabel_count[6] += 1
        elif ins_count < 5000:
            label2binlabel[7, cid] = binlabel_count[7]
            binlabel_count[7] += 1
        else:
            label2binlabel[8, cid] = binlabel_count[8]
            binlabel_count[8] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/8bins/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((9, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/8bins/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    pdb.set_trace()

def get_split8():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin10 = []
    bin100 = []
    bin1000 = []
    binover = []
    bin5 = []
    bin50 = []
    bin500 = []
    bin5000 = []


    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']

        if ins_count < 5:
            bin5.append(cid)
        elif ins_count < 10:
            bin10.append(cid)
        elif ins_count < 50:
            bin50.append(cid)
        elif ins_count < 100:
            bin100.append(cid)
        elif ins_count < 500:
            bin500.append(cid)
        elif ins_count < 1000:
            bin1000.append(cid)
        elif ins_count < 5000:
            bin5000.append(cid)
        else:
            binover.append(cid)

    splits = {}
    splits['(5, 10)'] = np.array(bin10, dtype=np.int)
    splits['[50, 100)'] = np.array(bin100, dtype=np.int)
    splits['[500, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[5000, ~)'] = np.array(binover, dtype=np.int)
    splits['(0, 5)'] = np.array(bin5, dtype=np.int)
    splits['[10, 50)'] = np.array(bin50, dtype=np.int)
    splits['[100, 500)'] = np.array(bin500, dtype=np.int)
    splits['[1000, 5000)'] = np.array(bin5000, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/8bins/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)


def get_draw_val_imgs():
    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    val_img_ann = lvis_val.img_ann_map

    bin100 = set()

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']

        if ins_count < 20:
            bin100.add(cid)

    print('bin100----------', bin100)
    print(len(bin100))
    with open(val_ann_file, 'r') as fin:
        data = json.load(fin)

    draw_val = {
        'info':data['info'],
        'annotations':data['annotations'],
        'categories': data['categories'],
        'licenses': data['licenses']
    }

    imglist = []
    for im in data['images']:
        id = im['id']
        catids = set([v['category_id'] for v in val_img_ann[id]])

        if len(catids & bin100) > 0:
            imglist.append(im)

    draw_val['images'] = imglist
    print('-------------', len(imglist))

    with open('./data/lvis/draw_val.json', 'w') as fout:
        json.dump(draw_val, fout)


def get_hist():
    train_ann_file = './data/lvis/lvis_v0.5_val.json'
    lvis_train = LVIS(train_ann_file)
    img_ann_map = lvis_train.img_ann_map

    hist = dict()
    for k, v in img_ann_map.items():
        ins_num = len(img_ann_map[k])
        if ins_num not in hist:
            hist[ins_num] = 1
        else:
            hist[ins_num] += 1

    with open('tempdata/lvis_hist_val.pkl', 'wb') as fout:
        pickle.dump(hist, fout)


def get_dense_det():
    train_ann_file = './data/lvis/lvis_v0.5_val.json'
    lvis_train = LVIS(train_ann_file)
    img_ann_map = lvis_train.img_ann_map

    set20 = set()
    set40 = set()
    set300 = set()
    for k, v in img_ann_map.items():
        ins_num = len(img_ann_map[k])
        if ins_num >= 20:
            set20.add(k)
        if ins_num >= 40:
            set40.add(k)
        if ins_num >= 300:
            set300.add(k)

    with open(train_ann_file, 'r') as fin:
        data = json.load(fin)
    # data: ['images', 'info', 'annotations', 'categories', 'licenses']

    ann20 = []
    ann40 = []
    for ann in data['annotations']:
        if ann['image_id'] in set20:
            ann20.append(ann)
        if ann['image_id'] in set40:
            ann40.append(ann)

    img20 = []
    img40 = []
    for im in data['images']:
        if im['id'] in set20:
            img20.append(im)
        if im['id'] in set40:
            img40.append(im)

    data_20 = {
        'images': img20,
        'info': data['info'],
        'annotations': ann20,
        'categories': data['categories'],
        'licenses': data['licenses']}
    save_path = './data/lvis/lvis_v0.5_val_20.json'
    with open(save_path, 'w') as fout:
        json.dump(data_20, fout)

    data_40 = {
        'images': img40,
        'info': data['info'],
        'annotations': ann40,
        'categories': data['categories'],
        'licenses': data['licenses']}
    save_path = './data/lvis/lvis_v0.5_val_40.json'
    with open(save_path, 'w') as fout:
        json.dump(data_40, fout)


def del_tail():

    train_ann_file = './data/lvis/lvis_v0.5_val.json'
    # val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    cats_head = set()
    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count > 100:
            cats_head.add(cid)

    with open(train_ann_file, 'r') as fin:
        traindata = json.load(fin)

    new_ann = []
    new_img_set = set()
    for ann in traindata['annotations']:
        cid = ann['category_id']
        if cid in cats_head:
            new_ann.append(ann)
            new_img_set.add(ann['image_id'])

    new_images = []
    for img in traindata['images']:
        if img['id'] in new_img_set:
            not_exhaustive_category_ids = img['not_exhaustive_category_ids']
            new_not = []
            for cc in not_exhaustive_category_ids:
                if cc in cats_head:
                    new_not.append(cc)
            img['not_exhaustive_category_ids'] = new_not
            neg_category_ids = img['neg_category_ids']
            new_neg = []
            for cc in neg_category_ids:
                if cc in cats_head:
                    new_neg.append(cc)
            img['neg_category_ids'] = new_neg
            new_images.append(img)

    new_cats = []
    for cat in traindata['categories']:
        if cat['id'] in cats_head:
            new_cats.append(cat)

    no_tail_data = {
        'images': new_images,
        'annotations': new_ann,
        'categories': new_cats,
        'info': traindata['info'],
        'licenses': traindata['licenses']
    }
    save_file = './data/lvis/lvis_v0.5_val_headonly.json'
    with open(save_file, 'w') as fout:
        json.dump(no_tail_data, fout)

def construct_data():

    train_ann_file = './data/lvis/lvis_v0.5_train_headonly.json'
    val_ann_file = './data/lvis/lvis_v0.5_val_headonly.json'

    lvis_train = LVIS(train_ann_file)
    lvis_val = LVIS(val_ann_file)

    img_ann_map_train = lvis_train.img_ann_map
    img_ann_map_val = lvis_val.img_ann_map

    train_train = []
    train_20 = []
    train_40 = []
    for k, v in img_ann_map_train.items():
        ins_num = len(img_ann_map_train[k])
        if ins_num < 20 or ins_num > 300:
            train_train.append(k)
        elif ins_num < 40:
            train_20.append(k)
        else:
            train_40.append(k)

    val_train = []
    val_20 = []
    val_40 = []
    for k, v in img_ann_map_val.items():
        ins_num = len(img_ann_map_val[k])
        if ins_num < 20 or ins_num > 300:
            val_train.append(k)
        elif ins_num < 40:
            val_20.append(k)
        else:
            val_40.append(k)

    train_new = train_train + val_train

    cat2img_20 = {}
    for im in train_20:
        anns = img_ann_map_train[im]
        for ann in anns:
            cid = ann['category_id']
            if cid in cat2img_20:
                cat2img_20[cid].add(im)
            else:
                cat2img_20[cid] = {im}

    train_val_20 = set()
    for cid, imgs in cat2img_20.items():
        img_num = len(imgs)
        sample_num = int(img_num / 2)
        rest_img = imgs - train_val_20
        already_got_num = img_num - len(rest_img)
        sample_num = sample_num - already_got_num
        if sample_num <= 0:
            continue
        choose = np.random.choice(len(rest_img), sample_num, replace=False)
        rest_img = list(rest_img)
        for i in choose:
            train_val_20.add(rest_img[int(i)])

    cat2img_40 = {}
    for im in train_40:
        anns = img_ann_map_train[im]
        for ann in anns:
            cid = ann['category_id']
            if cid in cat2img_40:
                cat2img_40[cid].add(im)
            else:
                cat2img_40[cid] = {im}
    train_val_40 = set()
    for cid, imgs in cat2img_40.items():
        img_num = len(imgs)
        sample_num = int(img_num / 2)
        rest_img = imgs - train_val_40
        already_got_num = img_num - len(rest_img)
        sample_num = sample_num - already_got_num
        if sample_num <= 0:
            continue
        choose = np.random.choice(len(rest_img), sample_num, replace=False)
        rest_img = list(rest_img)
        for i in choose:
            train_val_40.add(rest_img[int(i)])

    val_new_20 = val_20 + list(train_val_20)
    val_new_40 = val_40 + list(train_val_40)
    val_new = set(val_new_20 + val_new_40)
    train_new = set(train_new + list(set(train_20) - train_val_20) + \
                list(set(train_40) - train_val_40))

    with open(train_ann_file, 'r') as fin:
        traindata = json.load(fin)
    with open(val_ann_file, 'r') as fin:
        valdata = json.load(fin)

    train_img = []
    val_img = []
    for img in (traindata['images'] + valdata['images']):
        if img['id'] in train_new:
            train_img.append(img)
        elif img['id'] in val_new:
            val_img.append(img)
        else:
            print('NO SET ERROR! {}'.format(img))

    train_ann = []
    val_ann = []
    for ann in (traindata['annotations'] + valdata['annotations']):
        if ann['image_id'] in train_new:
            train_ann.append(ann)
        elif ann['image_id'] in val_new:
            val_ann.append(ann)
        else:
            print('ANN NO SET ERROR! {}'.format(ann))

    save_train = {
        'images': train_img,
        'annotations': train_ann,
        'categories': traindata['categories'],
        'info': traindata['info'],
        'licenses': traindata['licenses']
    }

    save_dir = './data/lvis/dense/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'dense_lvis_v0.5_train.json')
    with open(save_path, 'w') as fout:
        json.dump(save_train, fout)

    save_val = {
        'images': val_img,
        'annotations': val_ann,
        'categories': valdata['categories'],
        'info': valdata['info'],
        'licenses': valdata['licenses']
    }
    save_path = os.path.join(save_dir, 'dense_lvis_v0.5_val.json')
    with open(save_path, 'w') as fout:
        json.dump(save_val, fout)


def get_val():
    train_ann_file = './data/lvis/lvis_v0.5_train_headonly.json'
    val_ann_file = './data/lvis/lvis_v0.5_val_headonly.json'

    with open(train_ann_file, 'r') as fin:
        train_ori = json.load(fin)
    with open(val_ann_file, 'r') as fin:
        val_ori = json.load(fin)

    all_img = train_ori['images'] + val_ori['images']
    all_ann = train_ori['annotations'] + val_ori['annotations']

    train_final_file = './data/lvis/dense/dense_lvis_v0.5_train.json'
    with open(train_final_file, 'r') as fin:
        train_final = json.load(fin)

    train_set = set()
    for im in train_final['images']:
        train_set.add(im['id'])
    all_set = set()
    for im in all_img:
        all_set.add(im['id'])

    val_set = all_set - train_set
    val_img = []
    for im in all_img:
        if im['id'] in val_set:
            val_img.append(im)
    val_ann = []
    for ann in all_ann:
        if ann['image_id'] in val_set:
            val_ann.append(ann)

    valdata = val_ori
    pdb.set_trace()
    save_dir = './data/lvis/dense/'
    save_val = {
        'images': val_img,
        'annotations': val_ann,
        'categories': train_final['categories'],
        'info': valdata['info'],
        'licenses': valdata['licenses']
    }
    save_path = os.path.join(save_dir, 'dense_lvis_v0.5_val.json')
    with open(save_path, 'w') as fout:
        json.dump(save_val, fout)


def count_ins():

    train_ann_file = './data/lvis/dense_v3/train.json'
    # train_ann_file = './data/lvis/dense_v3/val.json'


    # For training set
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    counts = {}
    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        counts[cid] = ins_count

    pdb.set_trace()

    with open('tempdata/dense_train_catins.pkl', 'wb') as fout:
        pickle.dump(counts, fout)


def del_nondense_cls():
    train_ann_file = './data/lvis/dense/dense_lvis_v0.5_train.json'
    val_ann_file = './data/lvis/dense/dense_lvis_v0.5_val.json'

    with open(train_ann_file, 'r') as fin:
        train_old = json.load(fin)
    with open(val_ann_file, 'r') as fin:
        val = json.load(fin)


    lvisval = LVIS(val_ann_file)
    # valcats = lvisval.cats
    # pdb.set_trace()

    cid_set = set()
    for ann in val['annotations']:
        cid_set.add(ann['category_id'])

    img_set = set()
    ann_new_train = []
    for ann in train_old['annotations']:
        if ann['category_id'] in cid_set:
            ann_new_train.append(ann)
            img_set.add(ann['image_id'])
    img_new_train = []
    for img in train_old['images']:
        if img['id'] in img_set:
            img_new_train.append(img)

    new_cats = []
    for cat in train_old['categories']:
        if cat['id'] in cid_set:
            new_cats.append(cat)


    save_train = {
        'images': img_new_train,
        'annotations': ann_new_train,
        'categories': new_cats,
        'info': train_old['info'],
        'licenses': train_old['licenses']
    }

    save_val = val
    save_val['categories'] = new_cats

    pdb.set_trace()
    save_dir = './data/lvis/densenew/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'dense_lvis_v0.5_train.json')
    with open(save_path, 'w') as fout:
        json.dump(save_train, fout)

    save_path = os.path.join(save_dir, 'dense_lvis_v0.5_val.json')
    with open(save_path, 'w') as fout:
        json.dump(save_val, fout)


def update_cls():

    # data['categories'][0]
    # {'frequency': 'f', 'id': 3, 'synset': 'air_conditioner.n.01', 'image_count': 212, 'instance_count': 487, 'synonyms': ['air_conditioner'], 'def': 'a machine that keeps air cool and dry', 'name': 'air_conditioner'}
    train_ann_file = './data/lvis/densenew/dense_lvis_v0.5_val.json'
    # val_ann_file = './data/lvis/densenew/dense_lvis_v0.5_val.json'

    with open(train_ann_file, 'r') as fin:
        train_data = json.load(fin)

    image_count = {}
    instance_count = {}
    cat_set = set()
    for ann in train_data['annotations']:
        cid = ann['category_id']
        iid = ann['image_id']
        cat_set.add(cid)

        if cid in image_count:
            image_count[cid].add(iid)
        else:
            image_count[cid] = {iid}

        if cid in instance_count:
            instance_count[cid] += 1
        else:
            instance_count[cid] = 1

    pdb.set_trace()

    new_cat = []
    for cat in train_data['categories']:
        if cat['id'] in cat_set:
            cat['instance_count'] = instance_count[cat['id']]
            cat['image_count'] = len(image_count[cat['id']])
            new_cat.append(cat)

    train_data['categories'] = new_cat
    pdb.set_trace()

    save_dir = './data/lvis/densecat/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'dense_lvis_v0.5_val.json')
    with open(save_path, 'w') as fout:
        json.dump(train_data, fout)


if __name__ == '__main__':

    # ana_param()
    # get_mask()
    # trymapping()
    # ana_coco_param()
    # load_checkpoint()
    # get_cate_gs()
    # get_cate_weight()

    # get_bin_weight()

    # get_cate_gs2()
    # get_split2()

    # get_draw_val_imgs()
    # load_checkpoint_all()

    # get_cate_gs8()
    # get_split8()

    # get_cate_weight_bf()
    # get_cate_weight_bours()

    # get_hist()
    # get_dense_det()
    # del_tail()
    # construct_data()
    # get_val()
    count_ins()
    # del_nondense_cls()
    # update_cls()