"""Compare the trained Siamese embedding vs. the baseline ReID embedding.

Question: is the Siamese redundant given we already have a strong ReID model?
Both models embed the SAME held-out crops (S02 val, unseen by the Siamese);
embeddings are L2-normalized and compared with cosine (fair: identical images,
metric, and dimensionality).

Outputs:
  - t-SNE side-by-side plot colored by vehicle identity (the chosen deliverable)
  - a compact metrics table: cross-camera image & tracklet Rank-1/mAP, pairwise
    ROC-AUC, mean intra/inter-class cosine similarity (supporting numbers)

Usage:
  python compare_embeddings.py \
    --crops ../../datasets/siam/crops/val \
    --siamese-ckpt ../../datasets/siam/siamese_d2048.pth \
    --reid-weights /home/likef/workspace/AIC21-MTMC/reid/reid_model/resnet101_ibn_a_2.pth \
    --reid-backbone resnet101_ibn_a --out-dir ../../datasets/siam/compare
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))         # ablation/visualization -> repo root
sys.path.insert(0, os.path.join(REPO, 'reid'))                 # -> reid_inference
sys.path.insert(0, REPO)                                       # -> config
sys.path.insert(0, os.path.join(REPO, 'ablation', 'siamese'))  # -> siamese_model

from siamese_model import SiameseEmbedding

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def list_crops(crops_dir, max_per_id):
    """-> (paths, labels, cams).  filename: <scene>_<cam>_f<frame>_b<i>.jpg ; dir=<scene>_<id>."""
    paths, labels, cams = [], [], []
    for d in sorted(Path(crops_dir).iterdir()):
        if not d.is_dir():
            continue
        imgs = sorted(d.glob('*.jpg'))
        if max_per_id:
            imgs = imgs[:max_per_id]
        for p in imgs:
            cam = p.name.split('_')[1]
            paths.append(str(p)); labels.append(d.name); cams.append(cam)
    return paths, np.array(labels), np.array(cams)


@torch.no_grad()
def embed(model, paths, tf, device, batch=128):
    out = []
    for i in range(0, len(paths), batch):
        ims = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + batch]])
        f = model(ims.to(device))
        out.append(F.normalize(f, p=2, dim=1).cpu())
    return torch.cat(out).numpy()


def load_reid(weights, backbone, size, device):
    from config import cfg
    cfg.merge_from_file(os.path.join(REPO, 'config', 'aic_reid1.yml'))
    cfg.REID_MODEL = weights
    cfg.REID_BACKBONE = backbone
    cfg.REID_SIZE_TEST = [size, size]
    from reid_inference.reid_model import build_reid_model
    model, _ = build_reid_model(cfg)
    return model.to(device).eval()


def eval_retrieval(feats, labels, cams):
    """Cross-camera Rank-1 / mAP (gallery excludes same-camera, incl. self)."""
    sim = feats @ feats.T
    rank1, aps, valid = 0, [], 0
    for i in range(len(labels)):
        order = np.argsort(-sim[i])
        keep = cams[order] != cams[i]            # cross-camera gallery (drops self)
        g_lab = labels[order][keep]
        match = (g_lab == labels[i]).astype(np.int32)
        if match.sum() == 0:
            continue
        valid += 1
        rank1 += int(match[0])
        cum = np.cumsum(match)
        prec = cum / (np.arange(len(match)) + 1)
        aps.append((prec * match).sum() / match.sum())
    return (rank1 / max(valid, 1), float(np.mean(aps)) if aps else 0.0, valid)


def tracklet_pool(feats, labels, cams):
    """Mean-pool per (label, cam) tracklet -> (tfeats, tlabels, tcams)."""
    groups = {}
    for i, (l, c) in enumerate(zip(labels, cams)):
        groups.setdefault((l, c), []).append(i)
    tf_, tl, tc = [], [], []
    for (l, c), idx in groups.items():
        v = feats[idx].mean(0)
        v = v / (np.linalg.norm(v) + 1e-12)
        tf_.append(v); tl.append(l); tc.append(c)
    return np.array(tf_), np.array(tl), np.array(tc)


def pairwise_stats(feats, labels, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(labels)
    i = rng.integers(0, N, n); j = rng.integers(0, N, n)
    m = i != j
    i, j = i[m], j[m]
    s = (feats[i] * feats[j]).sum(1)
    same = (labels[i] == labels[j]).astype(np.int32)
    auc = roc_auc_score(same, s) if same.sum() and (1 - same).sum() else float('nan')
    return auc, float(s[same == 1].mean()), float(s[same == 0].mean())


def tsne_plot(emb_r, emb_s, labels, out_png, k_ids=12, seed=0):
    uniq, counts = np.unique(labels, return_counts=True)
    top = uniq[np.argsort(-counts)[:k_ids]]
    mask = np.isin(labels, top)
    lab = labels[mask]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, emb, name in [(axes[0], emb_r[mask], 'ReID (resnet101-ibn)'),
                          (axes[1], emb_s[mask], 'Siamese (ours)')]:
        xy = TSNE(n_components=2, init='pca', perplexity=30, random_state=seed).fit_transform(emb)
        for t in top:
            sel = lab == t
            ax.scatter(xy[sel, 0], xy[sel, 1], s=10, label=t)
        ax.set_title(f'{name} — t-SNE, top {k_ids} identities')
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(fontsize=6, ncol=2, markerscale=1.5, loc='best')
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', required=True)
    ap.add_argument('--siamese-ckpt', required=True)
    ap.add_argument('--reid-weights', required=True)
    ap.add_argument('--reid-backbone', default='resnet101_ibn_a')
    ap.add_argument('--reid-size', type=int, default=384)
    ap.add_argument('--siam-size', type=int, default=224)
    ap.add_argument('--out-dir', default='../../datasets/siam/compare')
    ap.add_argument('--max-per-id', type=int, default=0, help='0 = use all crops')
    ap.add_argument('--tsne-ids', type=int, default=12)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, cams = list_crops(args.crops, args.max_per_id)
    print(f'{len(paths)} crops, {len(set(labels))} identities, {len(set(cams))} cameras')

    # --- ReID embedding ---
    reid = load_reid(args.reid_weights, args.reid_backbone, args.reid_size, device)
    reid_tf = transforms.Compose([
        transforms.Resize((args.reid_size, args.reid_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    emb_r = embed(reid, paths, reid_tf, device)

    # --- Siamese embedding ---
    ck = torch.load(args.siamese_ckpt, map_location=device)
    siam = SiameseEmbedding(dim=ck.get('dim', 2048), freeze_backbone=True,
                            normalize=ck.get('normalize', True)).to(device).eval()
    siam.load_state_dict(ck['model'])
    siam_tf = transforms.Compose([
        transforms.Resize((args.siam_size, args.siam_size)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    emb_s = embed(siam, paths, siam_tf, device)

    # --- metrics ---
    rows = []
    for name, emb in [('ReID', emb_r), ('Siamese', emb_s)]:
        r1, mAP, v = eval_retrieval(emb, labels, cams)
        tfeat, tlab, tcam = tracklet_pool(emb, labels, cams)
        tr1, tmap, tv = eval_retrieval(tfeat, tlab, tcam)
        auc, intra, inter = pairwise_stats(emb, labels)
        rows.append((name, r1, mAP, tr1, tmap, auc, intra, inter, intra - inter))

    hdr = ('model', 'img_R1', 'img_mAP', 'trk_R1', 'trk_mAP', 'pair_AUC',
           'intra_sim', 'inter_sim', 'gap')
    lines = ['  '.join(f'{h:>9}' for h in hdr)]
    for row in rows:
        lines.append('  '.join([f'{row[0]:>9}'] + [f'{x:9.4f}' for x in row[1:]]))
    table = '\n'.join(lines)
    print('\n=== cross-camera comparison (S02 val, held out from Siamese) ===')
    print(table)
    (out_dir / 'metrics.txt').write_text(table + '\n')

    tsne_plot(emb_r, emb_s, labels, str(out_dir / 'tsne_compare.png'), k_ids=args.tsne_ids)
    print(f'\nplot -> {out_dir / "tsne_compare.png"}\nmetrics -> {out_dir / "metrics.txt"}')


if __name__ == '__main__':
    main()
