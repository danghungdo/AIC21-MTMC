"""Merge ReID features for S02 (single-model). Reads detect_reid1_s02, writes detect_merge_s02.
Run from reid/:  python ../ablation/s02_pipeline/merge_reid_feat_s02.py aic_reid_s02.yml
"""
import os
import pickle
import sys

import numpy as np
from sklearn import preprocessing

sys.path.append('../')
from config import cfg

CAMS = ['c006', 'c007', 'c008', 'c009']

# optional argv: [2]=reid subdir (extract output), [3]=merge target subdir
ENSEMBLE = [sys.argv[2]] if len(sys.argv) > 2 else ['detect_reid1_s02']
MERGE_NAME = sys.argv[3] if len(sys.argv) > 3 else 'detect_merge_s02'


def merge_feat(_cfg):
    all_feat_dir = _cfg.DATA_DIR.split('detect')[0]  # -> 'datasets/'
    for cam in CAMS:
        feat_dic_list = []
        for feat_mode in ENSEMBLE:
            f = os.path.join(all_feat_dir, feat_mode, cam, f'{cam}_dets_feat.pkl')
            feat_dic_list.append(pickle.load(open(f, 'rb')))
        merged = feat_dic_list[0].copy()
        for patch in merged:
            feats = np.array([d[patch]['feat'] for d in feat_dic_list])
            feats = preprocessing.normalize(feats, norm='l2', axis=1)
            merged[patch]['feat'] = np.mean(feats, axis=0)
        out_dir = os.path.join(all_feat_dir, MERGE_NAME, cam)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f'{cam}_dets_feat.pkl')
        pickle.dump(merged, open(out, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('save pickle in %s' % out)


if __name__ == '__main__':
    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    merge_feat(cfg)
