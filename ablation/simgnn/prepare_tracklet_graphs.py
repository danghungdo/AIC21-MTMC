"""Step 3c data prep: build LABELED tracklet graphs from train/val crops.

Each (identity, camera) group of crops = one single-camera tracklet. We embed every
crop with the baseline ReID model, then build a graph per tracklet (same rule as
graph_construction.py: nodes = L2-normalized ReID feats, edges = pairwise Euclidean
distance > tau). Output carries the identity label so train_simgnn.py can form
same/different cross-camera pairs.

Crops layout (from ablation/siamese/prepare_crops.py):
    <crops>/<scene>_<id>/<scene>_<cam>_f<frame>_b<box>.jpg
identity = "<scene>_<id>" ; camera = "<cam>".

Usage (from ablation/simgnn):
    python prepare_tracklet_graphs.py --crops ../../datasets/siam/crops/train \
        --reid-weights /home/likef/workspace/AIC21-MTMC/reid/reid_model/resnet101_ibn_a_2.pth \
        --out ../../datasets/siam/graphs_train.pkl --tau 0.5 --max-nodes 15
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from graph_construction import build_edges, sample_nodes

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))    # ablation/simgnn -> repo root
sys.path.insert(0, os.path.join(REPO, 'reid'))            # -> reid_inference
sys.path.insert(0, REPO)                                  # -> config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_reid(weights, backbone, size, device):
    from config import cfg
    cfg.merge_from_file(os.path.join(REPO, 'config', 'aic_reid1.yml'))
    cfg.REID_MODEL = weights
    cfg.REID_BACKBONE = backbone
    cfg.REID_SIZE_TEST = [size, size]
    from reid_inference.reid_model import build_reid_model
    model, _ = build_reid_model(cfg)
    return model.to(device).eval()


@torch.no_grad()
def embed(model, paths, tf, device, batch=128):
    out = []
    for i in range(0, len(paths), batch):
        ims = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + batch]])
        f = model(ims.to(device))
        out.append(torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', required=True)
    ap.add_argument('--reid-weights', required=True)
    ap.add_argument('--reid-backbone', default='resnet101_ibn_a')
    ap.add_argument('--reid-size', type=int, default=384)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--keep', choices=['greater', 'less'], default='greater')
    ap.add_argument('--max-nodes', type=int, default=15)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # group crops by (identity, camera)
    groups = defaultdict(list)
    for d in sorted(Path(args.crops).iterdir()):
        if not d.is_dir():
            continue
        identity = d.name
        for p in sorted(d.glob('*.jpg')):
            cam = p.name.split('_')[1]
            groups[(identity, cam)].append(str(p))
    print(f'{len(groups)} tracklets ((id,cam) groups) from {args.crops}')

    # embed all crops once
    all_paths = [p for ps in groups.values() for p in ps]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_reid(args.reid_weights, args.reid_backbone, args.reid_size, device)
    tf = transforms.Compose([
        transforms.Resize((args.reid_size, args.reid_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    feat_map = dict(zip(all_paths, embed(model, all_paths, tf, device)))

    tracklets = []
    for (identity, cam), paths in groups.items():
        feats = np.stack([feat_map[p] for p in paths]).astype(np.float32)
        feats, _ = sample_nodes(feats, list(range(len(feats))), args.max_nodes)
        edge_index, _, _ = build_edges(feats, args.tau, args.keep)
        tracklets.append({'identity': identity, 'cam': cam,
                          'node_feat': feats, 'edge_index': edge_index})

    by_id = defaultdict(set)
    for t in tracklets:
        by_id[t['identity']].add(t['cam'])
    multicam = sum(len(v) >= 2 for v in by_id.values())
    pickle.dump(tracklets, open(args.out, 'wb'))
    print(f'saved {len(tracklets)} tracklet graphs ({len(by_id)} identities, '
          f'{multicam} seen in >=2 cameras) -> {args.out}')


if __name__ == '__main__':
    main()
