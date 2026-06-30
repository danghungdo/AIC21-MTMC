"""S02 cross-camera matching (appearance-only) + IDF1 evaluation.

Replaces the S06-specific zone-based trajectory_fusion -> sub_cluster -> gen_res with a
clean appearance-only clusterer, fed the PREDICTED single-camera MOT tracklets from the
full pipeline (detection -> ReID -> MOT). For each (cam, track) it pools the per-detection
ReID features into a mean vector, clusters tracklets across cameras by cosine similarity,
assigns global IDs, writes a track3-format txt, and scores it with the official eval.py.

Run from ablation/s02_pipeline/:
    python cluster_and_eval_s02.py --thresholds 0.3 0.4 0.5 0.6 0.7
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_MOT_DIR = os.path.join(REPO, 'datasets', 'detect_merge_s02')
EVAL_DIR = os.path.join(REPO, 'datasets', 'mtmc_eval')
CAMS = ['c006', 'c007', 'c008', 'c009']
sys.path.insert(0, os.path.join(EVAL_DIR, 'eval'))


def load_tracklets(mot_dir):
    """(cam_int, tid) -> {'mean': (2048,) L2-normed, 'boxes': [(frame,x,y,w,h)]}."""
    tracklets = {}
    for cam in CAMS:
        cid = int(cam[1:])
        pkl = os.path.join(mot_dir, cam, f'{cam}_mot_feat.pkl')
        data = pickle.load(open(pkl, 'rb'))
        by_tid = defaultdict(lambda: {'feats': [], 'boxes': []})
        for rec in data.values():
            tid = int(rec['id'])
            x1, y1, x2, y2 = rec['bbox']
            frame = int(str(rec['frame']).replace('img', ''))
            by_tid[tid]['feats'].append(np.asarray(rec['feat'], dtype=np.float32))
            by_tid[tid]['boxes'].append((frame, int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        for tid, d in by_tid.items():
            m = np.mean(d['feats'], axis=0)
            tracklets[(cid, tid)] = {'mean': m / (np.linalg.norm(m) + 1e-12),
                                     'boxes': d['boxes']}
    return tracklets


def cluster(keys, tracklets, thr):
    if len(keys) == 1:
        return {keys[0]: 0}
    M = np.stack([tracklets[k]['mean'] for k in keys])
    sim = M @ M.T
    dist = np.clip(1 - sim, 0, 2)
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - thr,
                                     metric='precomputed', linkage='average').fit_predict(dist)
    return {k: int(l) for k, l in zip(keys, labels)}


def write_pred(path, tracklets, key_label):
    with open(path, 'w') as f:
        for (cid, tid), d in tracklets.items():
            gid = key_label[(cid, tid)]
            for (fr, x, y, w, h) in d['boxes']:
                f.write(f'{cid} {gid} {fr} {x} {y} {w} {h} -1 -1\n')


def score(pred_path):
    import eval as ev
    test = ev.readData(os.path.join(EVAL_DIR, 'gt_S02.txt'))
    pred = ev.readData(pred_path)
    s = ev.eval(test, pred, dstype='validation', roidir=os.path.join(EVAL_DIR, 'ROIs'))
    return (float(s['idf1'].iloc[0]) * 100, float(s['idp'].iloc[0]) * 100,
            float(s['idr'].iloc[0]) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument('--mot-dir', default=DEFAULT_MOT_DIR, help='dir with <cam>/<cam>_mot_feat.pkl')
    ap.add_argument('--out-dir', default=os.path.join(EVAL_DIR, 'preds_s02'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tracklets = load_tracklets(args.mot_dir)
    keys = list(tracklets.keys())
    ncam = len(set(c for c, _ in keys))
    print(f'S02 predicted tracklets: {len(keys)} over {ncam} cameras')

    for thr in args.thresholds:
        key_label = cluster(keys, tracklets, thr)
        pred_path = os.path.join(args.out_dir, f'pred_cosine_thr{thr}.txt')
        write_pred(pred_path, tracklets, key_label)
        nclu = len(set(key_label.values()))
        idf1, idp, idr = score(pred_path)
        print(f'[cosine thr={thr}]  clusters {nclu}  IDF1 {idf1:.2f}  IDP {idp:.2f}  IDR {idr:.2f}')


if __name__ == '__main__':
    main()
