import mmcv
import numpy as np
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from lvis.lvis import LVIS
from lvis import LVISEval

from .recall import eval_recalls
import pickle

##a wrapper around LVISEval

# TODO: using the config file ann path instead of a fix one.
ANNOTATION_PATH = "./data/lvis/lvis_v0.5_val.json"

def lvis_eval(result_files, result_types, lvis, max_dets=(100, 300, 1000), existing_json=None):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'proposal_fast_percat', 'bbox', 'segm', 'keypoints'
        ]

    if mmcv.is_str(lvis):
        lvis = LVIS(lvis)
    assert isinstance(lvis, LVIS)

    if result_types == ['proposal_fast']:
        ar = lvis_fast_eval_recall(result_files, lvis, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    elif result_types == ['proposal_fast_percat']:
        assert existing_json is not None
        per_cat_recall = {}
        for cat_id in range(1, 1231):
            ar = lvis_fast_eval_recall(result_files, lvis, np.array(max_dets), category_id=cat_id)
            for i, num in enumerate(max_dets):
                per_cat_recall.update({cat_id:ar})
                print('cat{} AR@{}\t= {:.4f}'.format(cat_id, num, ar[i]))
        pickle.dump(per_cat_recall, open('./{}_per_cat_recall.pt'.format(existing_json), 'wb'))
        return
    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        iou_type = 'bbox' if res_type == 'proposal' else res_type
        lvisEval = LVISEval(ANNOTATION_PATH, result_file, iou_type)
        # lvisEval.params.imgIds = img_ids
        if res_type == 'proposal':
            lvisEval.params.use_cats = 0
            lvisEval.params.max_dets = list(max_dets)

        lvisEval.run()
        lvisEval.print_results()


def lvis_fast_eval_recall(results,
                     lvis,
                     max_dets,
                     category_id=None,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
            format(type(results)))

    gt_bboxes = []
    img_ids = lvis.get_img_ids()
    for i in range(len(img_ids)):
        ann_ids = lvis.get_ann_ids(img_ids=[img_ids[i]])
        ann_info = lvis.load_anns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            # if ann.get('ignore', False) or ann['iscrowd']:
            #     continue
            if category_id:
                if ann.get('category_id') !=category_id:
                    continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=False)
    ar = recalls.mean(axis=1)
    return ar


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg, _ = results[idx]
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different score for det and segm
            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def results2json(dataset, results, out_file):
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
##add dumping proposal results
        json_results = proposal2json(dataset, np.stack([item[2] for item in results]))
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
        print('proposals dumped')
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files
