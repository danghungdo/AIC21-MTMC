"""Step 3c (online): train the SimGNN tracklet matcher.

Forms cross-camera pairs from labeled tracklet graphs (prepare_tracklet_graphs.py):
  - positive: two tracklets, SAME identity, DIFFERENT camera  -> label 1
  - negative: two tracklets, DIFFERENT identity               -> label 0
and trains SimGNNMatcher with BCE. Reports val ROC-AUC + accuracy@0.5.

Usage (from ablation/simgnn):
    python train_simgnn.py --train ../../datasets/siam/graphs_train.pkl \
        --val ../../datasets/siam/graphs_val.pkl \
        --epochs 10 --batch 128 --n-pos 8000 --n-neg 8000 \
        --out ../../datasets/siam/simgnn.pth
"""
import argparse
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from simgnn_model import SimGNNMatcher


def make_pairs(tracklets, n_pos, n_neg, seed):
    by_id = defaultdict(list)
    for i, t in enumerate(tracklets):
        by_id[t['identity']].append(i)
    multi_ids = [k for k, v in by_id.items()
                 if len({tracklets[i]['cam'] for i in v}) >= 2]
    ids = list(by_id)
    rng = random.Random(seed)
    pairs = []
    for _ in range(n_pos):
        members = by_id[rng.choice(multi_ids)]
        while True:
            a, b = rng.sample(members, 2)
            if tracklets[a]['cam'] != tracklets[b]['cam']:
                break
        pairs.append((a, b, 1.0))
    for _ in range(n_neg):
        i1 = rng.choice(ids)
        i2 = rng.choice(ids)
        while i2 == i1:
            i2 = rng.choice(ids)
        pairs.append((rng.choice(by_id[i1]), rng.choice(by_id[i2]), 0.0))
    rng.shuffle(pairs)
    return pairs


def to_tensors(tracklets, device):
    out = []
    for t in tracklets:
        x = torch.from_numpy(t['node_feat']).float().to(device)
        ei = torch.from_numpy(t['edge_index']).long().to(device)
        out.append((x, ei))
    return out


def run(model, pairs, tens, optimizer, device, batch, train):
    model.train(train)
    order = list(range(len(pairs)))
    if train:
        random.shuffle(order)
    scores, labels, loss_sum = [], [], 0.0
    for s in tqdm(range(0, len(order), batch), leave=False):
        idx = order[s:s + batch]
        if train:
            optimizer.zero_grad()
        batch_loss = 0.0
        for j in idx:
            a, b, y = pairs[j]
            xa, eia = tens[a]
            xb, eib = tens[b]
            with torch.set_grad_enabled(train):
                p = model(eia, xa, eib, xb)
            yt = torch.tensor([y], device=device)
            batch_loss = batch_loss + F.binary_cross_entropy(p, yt)
            scores.append(p.item())
            labels.append(y)
        if train:
            (batch_loss / len(idx)).backward()
            optimizer.step()
        loss_sum += float(batch_loss)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float('nan')
    acc = np.mean((np.array(scores) > 0.5) == (np.array(labels) > 0.5))
    return loss_sum / len(pairs), auc, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--n-pos', type=int, default=8000)
    ap.add_argument('--n-neg', type=int, default=8000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--input-dim', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--out', default='../../datasets/siam/simgnn.pth')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr = pickle.load(open(args.train, 'rb'))
    va = pickle.load(open(args.val, 'rb'))
    tr_t, va_t = to_tensors(tr, device), to_tensors(va, device)
    tr_pairs = make_pairs(tr, args.n_pos, args.n_neg, args.seed)
    va_pairs = make_pairs(va, args.n_pos // 4, args.n_neg // 4, args.seed + 1)
    print(f'device={device} train_tracklets={len(tr)} val_tracklets={len(va)} '
          f'train_pairs={len(tr_pairs)} val_pairs={len(va_pairs)}')

    model = SimGNNMatcher(input_dim=args.input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        tl, tauc, tacc = run(model, tr_pairs, tr_t, optimizer, device, args.batch, True)
        with torch.no_grad():
            vl, vauc, vacc = run(model, va_pairs, va_t, optimizer, device, args.batch, False)
        print(f'epoch {ep:02d}  train_loss {tl:.4f} auc {tauc:.4f} acc {tacc:.4f}  |  '
              f'val auc {vauc:.4f} acc {vacc:.4f}')
        if vauc >= best:
            best = vauc
            torch.save({'model': model.state_dict(), 'input_dim': args.input_dim,
                        'val_auc': vauc, 'epoch': ep}, args.out)
            print(f'  saved best (val_auc {vauc:.4f}) -> {args.out}')
    print(f'done. best val_auc={best:.4f}')


if __name__ == '__main__':
    main()
