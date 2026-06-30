# Ablation Study — Graph-Based Tracklet method vs. the strong-ReID baseline

This folder isolates our **rebuild and evaluation of the "Graph-Based Tracklet Features"
paper** (Nguyen et al., IEEE TMM 2023) on top of the AIC21 MTMC baseline. It is *not* part
of the production pipeline — it is the experimental study assessing whether the paper's
added components (Siamese embedding → graph construction → SimGNN matching) improve over
plain mean-feature cosine on the baseline's ReID features.

See `../PROGRESS_REPORT.md` for the full writeup and conclusions.

## Layout

| Subfolder | Paper step | Contents |
|-----------|-----------|----------|
| `siamese/` | **3a** — Siamese embedding | `siamese_model.py`, `prepare_crops.py`, `make_triplets.py`, `train_siamese.py`, `README.md` |
| `simgnn/` | **3b + 3c** — graph construction + graph matching + eval | `graph_construction.py`, `prepare_tracklet_graphs.py`, `simgnn_model.py`, `train_simgnn.py`, `eval_mtmc_validation.py` |
| `visualization/` | analysis | `compare_embeddings.py` (ReID vs Siamese), `visualize_reid_crosscam.py` (cross-camera t-SNE) |

All scripts compute the repo root from `__file__` and read shared assets from the repo
(`config/`, `reid/reid_inference/`, `reid/reid_model/`, `datasets/`, `SimGNN/`). Run each
from inside its own subfolder (relative `../../datasets/...` defaults assume that), conda
env `aic21-mtmc`. Each subfolder/file has usage in its header; `siamese/README.md` is the
fullest walkthrough.

## Pipeline / data flow

```
labeled crops (gt.txt+vdo.avi)
   │  siamese/prepare_crops.py
   ├──> siamese/  : ReID-feature crops ─ make_triplets ─ train_siamese ─> siamese_d2048.pth   (3a)
   │
   ├──> visualization/compare_embeddings.py : ReID vs Siamese (retrieval mAP, AUC, t-SNE)
   │    visualization/visualize_reid_crosscam.py : cross-camera embedding maps (color=id, marker=cam)
   │
   └──> simgnn/  : graph_construction (tracklet→graph, τ=0.5)        (3b)
                   prepare_tracklet_graphs (labeled graphs for train/val)
                   simgnn_model + train_simgnn (same/diff matcher)    (3c)
                   eval_mtmc_validation (official IDF1 on S02: oracle | cosine | simgnn)
```

## Headline findings (S02 validation; details in `../PROGRESS_REPORT.md`)

- Embedding retrieval (cross-camera mAP): baseline **ReID 0.99** vs our **Siamese 0.11**.
- MTMC matching IDF1 (GT tracklets, official scorer): **mean-cosine 99.0** vs **SimGNN 63.3** (oracle 100).
- ⇒ each added component is **redundant-or-worse** vs cosine on strong ReID features.

Caveats: prototype scale (113 train identities); SimGNN tested, not MGMN; the standard ReID
models are trained on the validation scenes (leak proven via Track2 `train_label.xml` cameras
c001–c040), so absolute scores are inflated — relative comparisons remain valid. The leak-free
retrain (all data now local) is the recommended next step.
