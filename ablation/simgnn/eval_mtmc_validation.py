"""Option A: end-to-end-ish MTMC IDF1 on the VALIDATION set (S02), using GT
single-camera tracklets so we measure cross-camera MATCHING quality in isolation.

Pipeline:
  GT tracklets (gt.txt, true global IDs STRIPPED) -> per-tracklet ReID feature
  -> cross-camera clustering (oracle | cosine | simgnn) -> assign predicted global IDs
  -> write track3-format prediction -> score with the official eval.py (IDF1).

The matcher is pluggable so baseline mean-cosine and SimGNN are directly comparable.
No crossroad zones (those are S06-only); clustering is appearance-based.

Usage (from ablation/simgnn):
  python eval_mtmc_validation.py --method cosine --thresholds 0.3 0.4 0.5 0.6 0.7
  python eval_mtmc_validation.py --method oracle          # harness sanity (~100 IDF1)
  python eval_mtmc_validation.py --method simgnn --simgnn-ckpt ../../datasets/siam/simgnn.pth --thresholds 0.5
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict
from glob import glob

import numpy as np
from sklearn.cluster import AgglomerativeClustering

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))  # ablation/simgnn -> repo root
DATA = os.path.join(REPO, 'datasets')
GT_DIR = os.path.join(DATA, 'siam_raw', 'validation', 'S02')
GRAPHS_VAL = os.path.join(DATA, 'siam', 'graphs_val.pkl')
EVAL_DIR = os.path.join(DATA, 'mtmc_eval')
sys.path.insert(0, os.path.join(EVAL_DIR, 'eval'))


def parse_gt_tracklets():
    """(cam_int, gtid) -> list of (frame, x, y, w, h) from S02 gt.txt files."""
    tracklets = defaultdict(list)
    for gt in sorted(glob(os.path.join(GT_DIR, 'c*', 'gt', 'gt.txt'))):
        cam = int(os.path.basename(os.path.dirname(os.path.dirname(gt)))[1:])
        for line in open(gt):
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            frame, gid = int(p[0]), int(p[1])
            x, y, w, h = int(float(p[2])), int(float(p[3])), int(float(p[4])), int(float(p[5]))
            tracklets[(cam, gid)].append((frame, x, y, w, h))
    return tracklets


def load_features(graphs_val=GRAPHS_VAL):
    """(cam_int, gtid) -> {'mean': (2048,), 'node_feat', 'edge_index'} from a graphs_val pkl."""
    feats = {}
    for g in pickle.load(open(graphs_val, 'rb')):
        gid = int(g['identity'].split('_')[1])
        cam = int(g['cam'][1:])
        m = g['node_feat'].mean(0)
        feats[(cam, gid)] = {'mean': m / (np.linalg.norm(m) + 1e-12),
                             'node_feat': g['node_feat'], 'edge_index': g['edge_index']}
    return feats


def cosine_sim_matrix(keys, feats):
    M = np.stack([feats[k]['mean'] for k in keys])
    return M @ M.T


def simgnn_sim_matrix(keys, feats, ckpt):
    import torch
    from simgnn_model import SimGNNMatcher
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(ckpt, map_location=dev)
    model = SimGNNMatcher(input_dim=ck['input_dim']).to(dev).eval()
    model.load_state_dict(ck['model'])
    tens = {k: (torch.from_numpy(feats[k]['node_feat']).float().to(dev),
                torch.from_numpy(feats[k]['edge_index']).long().to(dev)) for k in keys}
    n = len(keys)
    sim = np.eye(n)
    with torch.no_grad():
        for i in range(n):
            xa, ea = tens[keys[i]]
            for j in range(i + 1, n):
                xb, eb = tens[keys[j]]
                s = float(model(ea, xa, eb, xb).item())
                sim[i, j] = sim[j, i] = s
    return sim


def cluster_from_sim(sim, thr):
    dist = np.clip(1 - sim, 0, 2)
    return AgglomerativeClustering(n_clusters=None, distance_threshold=1 - thr,
                                   metric='precomputed', linkage='average').fit_predict(dist)


def write_pred(path, tracklets, key_label):
    """key_label: (cam,gid) -> predicted global id. Writes all GT boxes with predicted id."""
    with open(path, 'w') as f:
        for (cam, gid), boxes in tracklets.items():
            pid = key_label[(cam, gid)]
            for (fr, x, y, w, h) in boxes:
                f.write(f'{cam} {pid} {fr} {x} {y} {w} {h} -1 -1\n')


def score(pred_path):
    import eval as ev
    test = ev.readData(os.path.join(EVAL_DIR, 'gt_S02.txt'))
    pred = ev.readData(pred_path)
    summary = ev.eval(test, pred, dstype='validation', roidir=os.path.join(EVAL_DIR, 'ROIs'))
    return float(summary['idf1'].iloc[0]) * 100, float(summary['idp'].iloc[0]) * 100, float(summary['idr'].iloc[0]) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['oracle', 'cosine', 'simgnn'], default='cosine')
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument('--simgnn-ckpt', default=os.path.join(DATA, 'siam', 'simgnn.pth'))
    ap.add_argument('--graphs-val', default=GRAPHS_VAL,
                    help='tracklet-graph pkl with per-tracklet ReID node features (default: leaky baseline)')
    ap.add_argument('--out-dir', default=os.path.join(EVAL_DIR, 'preds'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tracklets = parse_gt_tracklets()
    feats = load_features(args.graphs_val)
    print(f'features from {args.graphs_val}')
    keys = list(tracklets.keys())
    have = [k for k in keys if k in feats]
    miss = [k for k in keys if k not in feats]
    print(f'S02: {len(keys)} GT tracklets ({len(set(c for c, _ in keys))} cameras); '
          f'features for {len(have)}, missing {len(miss)} (-> singletons)')

    if args.method == 'oracle':
        # true global id = gtid (sanity check; should score ~100)
        key_label = {(c, g): g for (c, g) in keys}
        write_pred(os.path.join(args.out_dir, 'pred_oracle.txt'), tracklets, key_label)
        idf1, idp, idr = score(os.path.join(args.out_dir, 'pred_oracle.txt'))
        print(f'[oracle]  IDF1 {idf1:.2f}  IDP {idp:.2f}  IDR {idr:.2f}')
        return

    sim = (cosine_sim_matrix(have, feats) if args.method == 'cosine'
           else simgnn_sim_matrix(have, feats, args.simgnn_ckpt))  # computed once
    next_id = max(g for _, g in keys) + 1
    for thr in args.thresholds:
        labels = cluster_from_sim(sim, thr)
        key_label = {k: int(l) for k, l in zip(have, labels)}
        for k in miss:                       # unmatched tracklets -> unique singleton ids
            key_label[k] = next_id; next_id += 1
        pred_path = os.path.join(args.out_dir, f'pred_{args.method}_thr{thr}.txt')
        write_pred(pred_path, tracklets, key_label)
        n_clusters = len(set(key_label.values()))
        idf1, idp, idr = score(pred_path)
        print(f'[{args.method} thr={thr}]  clusters {n_clusters}  IDF1 {idf1:.2f}  IDP {idp:.2f}  IDR {idr:.2f}')


if __name__ == '__main__':
    main()
