import logging
import os.path
import pickle
import numpy as np
from lvis import LVIS, LVISResults, LVISEval
import argparse

parser = argparse.ArgumentParser(description='MMDet test detector')
parser.add_argument('--boxjson', help='test config file path')
parser.add_argument('--segjson', type=str, default='None')
args = parser.parse_args()

def get_split_bin():
    split_file_name = './data/lvis/valsplit.pkl'

    if os.path.exists(split_file_name):
        with open(split_file_name, 'rb') as fin:
            splits = pickle.load(fin)
        print('Load split file from: {}'.format(split_file_name))
        return splits

def accumulate_acc(num_ins, num_get, splitbin):

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


# with open('tempcls.pkl', 'rb') as fin:
#     savelist = pickle.load(fin)

# num_get = savelist[0]
# num_ins = savelist[1]
# splitbin = get_split_bin()
# accumulate_acc(num_ins, num_get, splitbin)

# result and val files for 100 randomly sampled images.
ANNOTATION_PATH = "data/lvis/lvis_v0.5_val.json"

RESULT_PATH_BBOX = args.boxjson
print('Eval Bbox:')
ANN_TYPE = 'bbox'
lvis_eval = LVISEval(ANNOTATION_PATH, RESULT_PATH_BBOX, ANN_TYPE)
lvis_eval.run()
lvis_eval.print_results()

if not args.segjson == 'None':
    RESULT_PATH_SEGM = args.segjson
    print('Eval Segm:')
    ANN_TYPE = 'segm'
    lvis_eval = LVISEval(ANNOTATION_PATH, RESULT_PATH_SEGM, ANN_TYPE)
    lvis_eval.run()
    lvis_eval.print_results()
