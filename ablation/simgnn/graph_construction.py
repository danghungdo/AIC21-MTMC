"""GraphBased paper Step 3b: graph-based tracklet feature construction.

Turns each single-camera tracklet (a set of bbox embeddings) into a GRAPH:
  - nodes  = per-bbox embeddings (ReID features here; L2-normalized)
  - edges  = pairwise Euclidean distance between nodes, keeping edges whose
             distance satisfies the threshold rule (paper: keep distance > tau,
             tau=0.5; configurable because that tau was calibrated for the
             Siamese embedding, not ReID's scale)
  - edge_attr = the distance value

Consumes the per-camera trajectory pkls from trajectory_fusion.py
(exp/viz/<...>/trajectory/c0XX.pkl) and writes one graph file per camera. The
spatio-temporal metadata (cam, start/end zone, in/out time) is carried through so
Step 3c can apply the crossroad-zone filter before graph-similarity matching.

Usage (from ablation/simgnn):
  python graph_construction.py --traj-dir exp/viz/test/S06/trajectory \
      --out-dir exp/viz/test/S06/graphs --tau 0.5 --keep greater --max-nodes 15
"""
import argparse
import os
import pickle
from glob import glob

import numpy as np

# repo root + the baseline's trajectory output location (produced by trajectory_fusion.py)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_TOOLS_EXP = os.path.join(REPO, 'reid', 'reid-matching', 'tools', 'exp', 'viz', 'test', 'S06')


def load_tracklet_nodes(tracklet_entry, normalize):
    """Ordered per-bbox features for one tracklet -> (N, D) float32, plus frames."""
    inner = tracklet_entry['tracklet']
    fids = sorted(inner.keys())
    feats = np.stack([np.asarray(inner[f]['feat'], dtype=np.float32) for f in fids])
    if normalize:
        feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return feats, fids


def sample_nodes(feats, fids, max_nodes):
    """Evenly subsample to at most max_nodes (paper uses ~15 bboxes/tracklet)."""
    n = len(feats)
    if not max_nodes or n <= max_nodes:
        return feats, fids
    idx = np.linspace(0, n - 1, max_nodes).round().astype(int)
    return feats[idx], [fids[i] for i in idx]


def build_edges(feats, tau, keep):
    """Pairwise Euclidean distances -> undirected edge_index (2,E) + edge_attr (E,).

    keep='greater' -> edges with dist > tau ; keep='less' -> dist < tau.
    Returns edges (both directions for message passing) and the upper-triangle
    distance vector (for stats)."""
    n = len(feats)
    diff = feats[:, None, :] - feats[None, :, :]
    dist = np.linalg.norm(diff, axis=2)            # (n, n)
    iu, ju = np.triu_indices(n, k=1)
    d = dist[iu, ju]
    mask = d > tau if keep == 'greater' else d < tau
    ei, ej, ea = iu[mask], ju[mask], d[mask]
    # undirected: add both directions
    edge_index = np.concatenate([np.stack([ei, ej]), np.stack([ej, ei])], axis=1).astype(np.int64)
    edge_attr = np.concatenate([ea, ea]).astype(np.float32)
    return edge_index, edge_attr, d


def build_graph(entry, tau, keep, max_nodes, normalize):
    feats, fids = load_tracklet_nodes(entry, normalize)
    feats, fids = sample_nodes(feats, fids, max_nodes)
    edge_index, edge_attr, d_all = build_edges(feats, tau, keep)
    io_time = entry.get('io_time', [None, None])
    zone_list = entry.get('zone_list', [])
    graph = {
        'cam': entry['cam'],
        'tid': entry['tid'],
        'num_nodes': len(feats),
        'node_feat': feats,                       # (N, D) float32
        'edge_index': edge_index,                 # (2, E) int64
        'edge_attr': edge_attr,                   # (E,) float32 distances
        'mean_feat': np.asarray(entry['mean_feat'], dtype=np.float32),
        'io_time': io_time,
        'start_zone': zone_list[0] if zone_list else None,
        'end_zone': zone_list[-1] if zone_list else None,
        'frame_list': fids,
    }
    return graph, d_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj-dir', default=os.path.join(_TOOLS_EXP, 'trajectory'))
    ap.add_argument('--out-dir', default=os.path.join(_TOOLS_EXP, 'graphs'))
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--keep', choices=['greater', 'less'], default='greater',
                    help="paper keeps dist > tau ('greater')")
    ap.add_argument('--max-nodes', type=int, default=15, help='0 = use all bboxes')
    ap.add_argument('--no-normalize', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pkls = sorted(glob(os.path.join(args.traj_dir, '*.pkl')))
    if not pkls:
        raise SystemExit(f'no trajectory pkls in {args.traj_dir}')

    all_d, node_counts, edge_counts, empty = [], [], [], 0
    n_tracklets = 0
    for pkl in pkls:
        cam_traj = pickle.load(open(pkl, 'rb'))
        graphs = {}
        for key, entry in cam_traj.items():
            g, d_all = build_graph(entry, args.tau, args.keep,
                                   args.max_nodes, not args.no_normalize)
            graphs[(g['cam'], g['tid'])] = g
            n_tracklets += 1
            node_counts.append(g['num_nodes'])
            edge_counts.append(g['edge_index'].shape[1] // 2)
            empty += int(g['edge_index'].shape[1] == 0)
            all_d.append(d_all)
        out = os.path.join(args.out_dir, os.path.basename(pkl).replace('.pkl', '_graphs.pkl'))
        pickle.dump(graphs, open(out, 'wb'))
        print(f'{os.path.basename(pkl)}: {len(graphs)} graphs -> {out}')

    d = np.concatenate(all_d)
    pct = np.percentile(d, [5, 25, 50, 75, 95])
    print('\n=== Step 3b summary ===')
    print(f'tracklets/graphs: {n_tracklets}')
    print(f'nodes per graph : mean {np.mean(node_counts):.1f}  (max-nodes cap = {args.max_nodes})')
    print(f'edges per graph : mean {np.mean(edge_counts):.1f}  | empty-edge graphs: {empty} ({100*empty/n_tracklets:.1f}%)')
    print(f'intra-tracklet pairwise Euclidean distance percentiles (normalized feats):')
    print(f'   5% {pct[0]:.3f}  25% {pct[1]:.3f}  50% {pct[2]:.3f}  75% {pct[3]:.3f}  95% {pct[4]:.3f}')
    frac_gt = float((d > args.tau).mean())
    print(f'fraction of intra-tracklet pairs with dist > {args.tau}: {frac_gt:.3f}  '
          f'(= edges kept with keep="greater")')
    print('NOTE: tau=0.5 was the paper\'s value for Siamese embeddings; check the '
          'percentiles above to decide whether it suits these ReID features.')


if __name__ == '__main__':
    main()
