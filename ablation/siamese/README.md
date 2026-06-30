# Ablation 3a — Siamese embedding (GraphBased paper, Step 3a)

Part of the **ablation study** (`ablation/`). Trains the paper's Siamese embedding —
**ResNet-50 (frozen, ImageNet-pretrained) + 2 dense layers → d-dim embedding**,
optimized with **triplet loss** over offline triplets — producing a
Euclidean-metric-calibrated feature for graph Step 3b/3c (`../simgnn/`).

Related: `../visualization/` compares this Siamese embedding vs. the baseline ReID
embedding (the experiment that found the Siamese redundant — see `ablation/README.md`).

## Files
| File | Role |
|------|------|
| `prepare_crops.py` | `gt.txt` + `vdo.avi` → ID-labeled vehicle crops |
| `make_triplets.py` | crops → fixed list of 100k offline (anchor, positive, negative) triplets |
| `siamese_model.py` | the ResNet-50 + 2-dense-layer embedding network |
| `train_siamese.py` | triplet-loss training loop (Adam, frozen backbone) + val triplet-accuracy |

## Data (current prototype)
- Source: `datasets/siam_raw/{train,validation}/<scene>/<cam>/{vdo.avi,gt/gt.txt}`
  (train = S01+S03; val = S02, plus S05 later extracted — from the CityFlow train/val tarballs).
- Identity label = `<scene>_<id>`; AIC21 MTMC GT IDs are consistent across cameras
  within a scene, so one label = the same vehicle seen by multiple cameras.

## Run (from this dir, conda env `aic21-mtmc`)
```bash
# 1. crops  (per-id stride subsampling keeps counts manageable)
python prepare_crops.py --raw ../../datasets/siam_raw/train      --split train --stride 5
python prepare_crops.py --raw ../../datasets/siam_raw/validation --split val   --stride 5

# 2. offline triplets (paper: 100k)
python make_triplets.py --crops ../../datasets/siam/crops/train --n 100000 --out ../../datasets/siam/triplets_train.txt --seed 21
python make_triplets.py --crops ../../datasets/siam/crops/val   --n 10000  --out ../../datasets/siam/triplets_val.txt   --seed 7

# 3. train
python train_siamese.py \
  --triplets-train ../../datasets/siam/triplets_train.txt \
  --triplets-val   ../../datasets/siam/triplets_val.txt \
  --dim 2048 --margin 0.3 --epochs 15 --batch 64 --lr 3e-4 --workers 8 \
  --out ../../datasets/siam/siamese_d2048.pth
```
Add `--max-steps 20` to step 3 for a quick smoke test. `--no-normalize` disables L2-norm
of embeddings (default on, which bounds distances so the graph τ=0.5 is meaningful).

## Prototype dataset stats (as built)
- train: 9,106 crops / 113 identities · val: 4,361 crops / 145 identities.

## Notes / faithfulness to paper
- **Backbone frozen**, only the 2 dense head layers train (paper Step 3a). BN stats
  frozen during training.
- **Triplet loss** via `pytorch-metric-learning` on the fixed offline triplets ≡ paper's
  `L = max(0, m + ‖v_a−v_p‖ − ‖v_a−v_n‖)`.
- **Offline 100k triplets** (paper text), not online batch-hard mining.
- d=2048 is the paper's best; `--dim` lets you ablate 256/512/1024/2048 (paper Table VII).

## Scaling up beyond the prototype
- Add scenes: extract S04 (train) and S05 (val) from the tarballs into `datasets/siam_raw/`
  and re-run steps 1–3 (more identities → better embedding).
- Validate quality beyond triplet-accuracy: a Rank-1 / mAP ReID eval on a held-out
  camera split, and re-tune τ for Step 3b.

## Next (Step 3b/3c) — built in `../simgnn/`
The trained `siamese_d2048.pth` can feed the graph construction (`../simgnn/graph_construction.py`):
per tracklet, nodes = embeddings, edges = Euclidean distance > τ=0.5, then graph similarity
(SimGNN). NOTE: our current `../simgnn/` pipeline builds graphs on **ReID** features (not these
Siamese features), because the comparison in `../visualization/` found the Siamese redundant.
To run the paper-faithful variant, point `prepare_tracklet_graphs.py` embeddings at this Siamese model.
