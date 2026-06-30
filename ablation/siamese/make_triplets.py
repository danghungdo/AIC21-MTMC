"""Step 3a: pre-sample a fixed list of offline triplets (anchor, positive, negative).

Faithful to the paper, which "extracted 100,000 triplets (x_a, x_p, x_n)" as input to
the Siamese network. Each line: <anchor_path>\\t<positive_path>\\t<negative_path>
- anchor & positive: two different crops of the SAME identity
- negative: a crop of a DIFFERENT identity

Usage:
    python make_triplets.py --crops ../../datasets/siam/crops/train --n 100000 \
        --out ../../datasets/siam/triplets_train.txt --seed 21
    python make_triplets.py --crops ../../datasets/siam/crops/val   --n 10000  \
        --out ../../datasets/siam/triplets_val.txt   --seed 7
"""
import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', required=True, help='<out>/crops/<split> dir of <id>/*.jpg')
    ap.add_argument('--n', type=int, default=100000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=21)
    args = ap.parse_args()

    crops = Path(args.crops)
    id2imgs = {d.name: [str(p) for p in d.glob('*.jpg')]
               for d in crops.iterdir() if d.is_dir()}
    # need >=2 images to form an (anchor, positive) pair
    ids = [i for i, v in id2imgs.items() if len(v) >= 2]
    if len(ids) < 2:
        raise SystemExit(f'need >=2 identities with >=2 imgs each; got {len(ids)}')

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for _ in range(args.n):
            aid = rng.choice(ids)
            a, p = rng.sample(id2imgs[aid], 2)
            nid = rng.choice(ids)
            while nid == aid:
                nid = rng.choice(ids)
            n = rng.choice(id2imgs[nid])
            f.write(f'{a}\t{p}\t{n}\n')
    print(f'wrote {args.n} triplets from {len(ids)} identities -> {args.out}')


if __name__ == '__main__':
    main()
