"""Visualize ReID embeddings of vehicles ACROSS cameras.

Embeds GT-labeled vehicle crops with the baseline ReID model and plots a 2D t-SNE
where color = vehicle identity and marker = camera. Read it as:
  - same color, different markers, one tight cluster -> embedding is discriminative
    AND camera-invariant (good for cross-camera matching);
  - a vehicle splitting into per-camera blobs -> camera bias.

Also prints cross-camera consistency numbers: mean cosine between SAME-vehicle crops
from DIFFERENT cameras vs mean cosine between DIFFERENT vehicles.

Usage (from ablation/visualization):
  python visualize_reid_crosscam.py --crops ../../datasets/siam/crops/train --scene S03 \
      --reid-weights /home/likef/workspace/AIC21-MTMC/reid/reid_model/resnet101_ibn_a_2.pth \
      --k-ids 12 --out ../../datasets/siam/reid_crosscam_S03.png
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))         # ablation/visualization -> repo root
sys.path.insert(0, os.path.join(REPO, 'reid'))                 # -> reid_inference
sys.path.insert(0, REPO)                                       # -> config
sys.path.insert(0, os.path.join(REPO, 'ablation', 'siamese'))  # -> siamese_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>', 'p', 'h']


def load_reid(weights, backbone, size, device):
    from config import cfg
    cfg.merge_from_file(os.path.join(REPO, 'config', 'aic_reid1.yml'))
    cfg.REID_MODEL = weights
    cfg.REID_BACKBONE = backbone
    cfg.REID_SIZE_TEST = [size, size]
    from reid_inference.reid_model import build_reid_model
    model, _ = build_reid_model(cfg)
    return model.to(device).eval()


def load_siamese(ckpt_path, device):
    import torch as _torch
    from siamese_model import SiameseEmbedding
    ck = _torch.load(ckpt_path, map_location=device)
    model = SiameseEmbedding(dim=ck.get('dim', 2048), freeze_backbone=True,
                             normalize=ck.get('normalize', True)).to(device).eval()
    model.load_state_dict(ck['model'])
    return model


@torch.no_grad()
def embed(model, paths, tf, device, batch=128):
    out = []
    for i in range(0, len(paths), batch):
        ims = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + batch]])
        f = model(ims.to(device))
        out.append(F.normalize(f, p=2, dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', required=True)
    ap.add_argument('--scene', default='S03', help='scene prefix to filter (e.g. S03); ALL = no filter')
    ap.add_argument('--model', choices=['reid', 'siamese'], default='reid')
    ap.add_argument('--reid-weights', default=os.path.join(REPO, 'reid', 'reid_model', 'resnet101_ibn_a_2.pth'))
    ap.add_argument('--reid-backbone', default='resnet101_ibn_a')
    ap.add_argument('--reid-size', type=int, default=384)
    ap.add_argument('--siamese-ckpt', default=os.path.join(REPO, 'datasets', 'siam', 'siamese_d2048.pth'))
    ap.add_argument('--siam-size', type=int, default=224)
    ap.add_argument('--k-ids', type=int, default=12, help='# identities to show (most cross-camera)')
    ap.add_argument('--max-per-cam', type=int, default=8, help='cap crops per (id,cam) for readability')
    ap.add_argument('--out', default='../../datasets/siam/reid_crosscam.png')
    args = ap.parse_args()

    # collect crops grouped by identity, with camera
    by_id_cam = defaultdict(lambda: defaultdict(list))
    for d in sorted(Path(args.crops).iterdir()):
        if not d.is_dir() or (args.scene != 'ALL' and not d.name.startswith(args.scene + '_')):
            continue
        for p in sorted(d.glob('*.jpg')):
            cam = p.name.split('_')[1]
            by_id_cam[d.name][cam].append(str(p))

    # pick identities spanning the most cameras (most informative for cross-cam check)
    ranked = sorted(by_id_cam, key=lambda i: (len(by_id_cam[i]), sum(len(v) for v in by_id_cam[i].values())),
                    reverse=True)
    chosen = ranked[:args.k_ids]
    cams_all = sorted({c for i in chosen for c in by_id_cam[i]})
    print(f'scene={args.scene}: {len(by_id_cam)} identities; showing top {len(chosen)} '
          f'spanning cameras {cams_all}')

    paths, ids, cams = [], [], []
    for i in chosen:
        for c, ps in by_id_cam[i].items():
            for p in ps[:args.max_per_cam]:
                paths.append(p); ids.append(i); cams.append(c)
    ids, cams = np.array(ids), np.array(cams)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'reid':
        model = load_reid(args.reid_weights, args.reid_backbone, args.reid_size, device)
        size, interp = args.reid_size, transforms.InterpolationMode.BICUBIC
        model_title = 'ReID (resnet101-ibn)'
    else:
        model = load_siamese(args.siamese_ckpt, device)
        size, interp = args.siam_size, transforms.InterpolationMode.BILINEAR
        model_title = 'Siamese (frozen ResNet-50 + triplet head)'
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=interp),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    emb = embed(model, paths, tf, device)

    # --- cross-camera consistency numbers ---
    sim = emb @ emb.T
    same_id = ids[:, None] == ids[None, :]
    diff_cam = cams[:, None] != cams[None, :]
    iu = np.triu_indices(len(ids), 1)
    sid, dcam = same_id[iu], diff_cam[iu]
    s = sim[iu]
    intra_xcam = s[sid & dcam].mean()          # same vehicle, different camera
    inter = s[~sid].mean()                       # different vehicles
    print(f'mean cosine  same-vehicle/cross-camera: {intra_xcam:.3f}')
    print(f'mean cosine  different-vehicle        : {inter:.3f}')
    print(f'separation gap: {intra_xcam - inter:.3f}  (larger = more discriminative across cameras)')

    # --- t-SNE plot: color=identity, marker=camera ---
    xy = TSNE(n_components=2, init='pca', perplexity=min(30, len(ids) - 1), random_state=0).fit_transform(emb)
    cmap = plt.get_cmap('tab20')
    id_color = {i: cmap(k % 20) for k, i in enumerate(chosen)}
    cam_marker = {c: MARKERS[k % len(MARKERS)] for k, c in enumerate(cams_all)}

    fig, ax = plt.subplots(figsize=(11, 9))
    for i in chosen:
        for c in cams_all:
            m = (ids == i) & (cams == c)
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], color=id_color[i], marker=cam_marker[c],
                           s=55, edgecolors='k', linewidths=0.3, alpha=0.85)
    ax.set_title(f'{model_title} embeddings across cameras — {args.scene}\n'
                 f'color = vehicle ID, marker = camera  '
                 f'(same color+one cluster across markers = camera-invariant)')
    ax.set_xticks([]); ax.set_yticks([])
    cam_leg = [Line2D([0], [0], marker=cam_marker[c], color='w', markerfacecolor='gray',
                      markeredgecolor='k', markersize=9, label=c) for c in cams_all]
    id_leg = [Patch(facecolor=id_color[i], label=i) for i in chosen]
    l1 = ax.legend(handles=cam_leg, title='camera', loc='upper left', fontsize=8)
    ax.add_artist(l1)
    ax.legend(handles=id_leg, title='vehicle id', loc='upper right', fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(args.out, dpi=140); plt.close(fig)
    print(f'plot -> {args.out}')


if __name__ == '__main__':
    main()
