"""Step 3a: train the Siamese embedding with triplet loss on offline triplets.

Loss is pytorch-metric-learning's TripletMarginLoss applied to the fixed (a,p,n)
triplets (equivalent to the paper's L = max(0, m + ||v_a-v_p|| - ||v_a-v_n||)).
Backbone is frozen; only the two dense head layers train (Adam).

Usage:
    python train_siamese.py \
        --triplets-train ../../datasets/siam/triplets_train.txt \
        --triplets-val   ../../datasets/siam/triplets_val.txt \
        --dim 2048 --margin 0.3 --epochs 20 --batch 32 \
        --out ../../datasets/siam/siamese_d2048.pth
"""
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_metric_learning.losses import TripletMarginLoss
from tqdm import tqdm

from siamese_model import SiameseEmbedding

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TripletDataset(Dataset):
    def __init__(self, triplet_file, img_size, train=True):
        self.triplets = [ln.strip().split('\t') for ln in open(triplet_file) if ln.strip()]
        aug = [transforms.RandomHorizontalFlip()] if train else []
        self.tf = transforms.Compose(
            [transforms.Resize((img_size, img_size))] + aug +
            [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.triplets)

    def _load(self, p):
        return self.tf(Image.open(p).convert('RGB'))

    def __getitem__(self, i):
        a, p, n = self.triplets[i]
        return self._load(a), self._load(p), self._load(n)


def run_epoch(model, loader, loss_fn, optimizer, device, train):
    model.train(train)
    if train and model.freeze_backbone:
        model.backbone.eval()  # keep frozen BN stats fixed
    total_loss, correct, count = 0.0, 0, 0
    for a, p, n in tqdm(loader, leave=False):
        imgs = torch.cat([a, p, n], 0).to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            emb = model(imgs)
        B = emb.shape[0] // 3
        ai = torch.arange(0, B, device=device)
        pi = torch.arange(B, 2 * B, device=device)
        ni = torch.arange(2 * B, 3 * B, device=device)
        if train:
            labels = torch.zeros(3 * B, device=device)  # ignored when indices_tuple given
            loss = loss_fn(emb, labels, indices_tuple=(ai, pi, ni))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * B
        # triplet accuracy: d(a,p) < d(a,n)
        d_ap = F.pairwise_distance(emb[ai], emb[pi])
        d_an = F.pairwise_distance(emb[ai], emb[ni])
        correct += (d_ap < d_an).sum().item()
        count += B
    return total_loss / max(count, 1), correct / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--triplets-train', required=True)
    ap.add_argument('--triplets-val', required=True)
    ap.add_argument('--dim', type=int, default=2048)
    ap.add_argument('--margin', type=float, default=0.3)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=32, help='triplets per batch (x3 imgs)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-normalize', action='store_true', help='disable L2-norm of embeddings')
    ap.add_argument('--max-steps', type=int, default=0, help='>0: cap steps/epoch (smoke test)')
    ap.add_argument('--out', default='../../datasets/siam/siamese.pth')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SiameseEmbedding(dim=args.dim, freeze_backbone=True,
                             normalize=not args.no_normalize).to(device)
    loss_fn = TripletMarginLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)

    tr = TripletDataset(args.triplets_train, args.img_size, train=True)
    va = TripletDataset(args.triplets_val, args.img_size, train=False)
    if args.max_steps:
        from torch.utils.data import Subset
        tr = Subset(tr, range(min(len(tr), args.max_steps * args.batch)))
        va = Subset(va, range(min(len(va), args.max_steps * args.batch)))
        va.freeze_backbone = True
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f'device={device} train_triplets={len(tr)} val_triplets={len(va)} dim={args.dim}')

    best = 0.0
    for ep in range(1, args.epochs + 1):
        loss, acc = run_epoch(model, tl, loss_fn, optimizer, device, train=True)
        with torch.no_grad():
            _, vacc = run_epoch(model, vl, loss_fn, optimizer, device, train=False)
        print(f'epoch {ep:02d}  train_loss {loss:.4f}  train_acc {acc:.4f}  val_acc {vacc:.4f}')
        if vacc >= best:
            best = vacc
            torch.save({'model': model.state_dict(), 'dim': args.dim,
                        'normalize': not args.no_normalize, 'val_acc': vacc, 'epoch': ep}, args.out)
            print(f'  saved best (val_acc {vacc:.4f}) -> {args.out}')
    print(f'done. best val_acc={best:.4f}')


if __name__ == '__main__':
    main()
