# Ablation & Evaluation Workspace

Experimental work on top of the AIC21 MTMC baseline — **not** part of the production pipeline.
It now holds three related threads:

1. **Graph-method ablation** — rebuild & evaluate the *"Graph-Based Tracklet Features"* paper
   (Nguyen et al., IEEE TMM 2023): Siamese embedding → graph construction → SimGNN matching, vs.
   plain mean-feature cosine on the baseline's ReID features. (`siamese/`, `simgnn/`, `visualization/`)
2. **Leak-free S02 evaluation** — end-to-end MTMC on the held-out S02 scene with a leak-free ReID
   model, appearance-only cross-camera matching, scored by the official IDF1. (`s02_pipeline/`)
3. **S06 ensemble submission** — the faithful 3-model-ensemble baseline run that produced the
   official **IDF1 0.8090** on S06. (`s06_ensemble/`)

> Full writeups live in the global hub `~/workspace/thesis-docs/`:
> `baseline-aic21/{PROGRESS_REPORT,S02_EVAL_REPORT,REALTIME_PROFILE,NEW_DATASET_PLAN}.md`,
> `submissions/README.md`. This README is the in-repo map of the code.

## Layout

| Subfolder | Role | Key contents |
|-----------|------|--------------|
| `siamese/` | Paper **3a** — Siamese embedding | `siamese_model.py`, `prepare_crops.py`, `make_triplets.py`, `train_siamese.py`, `README.md` |
| `simgnn/` | Paper **3b+3c** — graph construction + SimGNN + matching-only eval | `graph_construction.py`, `prepare_tracklet_graphs.py`, `simgnn_model.py`, `train_simgnn.py`, `eval_mtmc_validation.py` |
| `visualization/` | analysis & qualitative | `compare_embeddings.py` (ReID vs Siamese), `visualize_reid_crosscam.py` (cross-cam t-SNE), `viz_grid_gt.py` (S02 GT grid clip) |
| `s02_pipeline/` | leak-free end-to-end S02 eval | `gen_images_s02.py`, `gen_det_s02.sh`, `merge_reid_feat_s02.py`, `run_aic_s02.sh`, `cluster_and_eval_s02.py`, `profile_pipeline.py`, `RESUME.md` |
| `s06_ensemble/` | S06 3-model ensemble for submission | `run_s06_ensemble.sh`, `run_aic_s06_serial.sh`, `validate_submission.py` |

All scripts compute the repo root from `__file__` and read shared assets (`config/`,
`reid/reid_inference/`, `reid/reid_model/`, `datasets/`, `SimGNN/`). Run each from inside its own
subfolder (relative `../../datasets/...` defaults assume that), conda env `aic21-mtmc`. Each
file's header has usage; `siamese/README.md` and `s02_pipeline/RESUME.md` are the fullest walkthroughs.

`conda run` buffers stdout — track long runs via the filesystem / task-completion, not the log tail.

## Data flow

```
1. Graph-method ablation (S02, GT tracklets — "matching-only" protocol)
   labeled crops (gt.txt+vdo.avi)
      │ siamese/prepare_crops.py
      ├─ siamese/  : crops ─ make_triplets ─ train_siamese ─> siamese_d2048.pth          (3a)
      ├─ visualization/ : compare_embeddings (ReID vs Siamese), visualize_reid_crosscam
      └─ simgnn/   : graph_construction (τ=0.5) ─ prepare_tracklet_graphs ─ train_simgnn  (3b/3c)
                     eval_mtmc_validation : official IDF1 (oracle | cosine | simgnn)

2. Leak-free S02 full pipeline ("full-pipeline" protocol)
   S02 video → gen_images_s02 → gen_det_s02 (YOLOv5x) → reid feats (leak-free _80) →
   merge_reid_feat_s02 → run_aic_s02 (MOT) → cluster_and_eval_s02 (appearance-only) → IDF1

3. S06 ensemble (official submission)
   S06 crops (reused) → reid2/reid3 → merge(3-model) → MOT(serial) → trajectory→sub_cluster→gen_res
   → track3.txt → validate_submission → submissions/s06_ensemble/track1.zip → 2022 Track1 server
```

## Headline findings

**Graph method is redundant-or-worse vs cosine on strong ReID** (S02, GT tracklets, official scorer):
- Embedding cross-camera mAP: baseline **ReID 0.99** vs **Siamese 0.11**.
- Matching IDF1: **mean-cosine 99.0** vs **SimGNN 63.3** (oracle 100) — but on *saturated, leaked* ReID.

**Leak-free, end-to-end (S02)** — see `S02_EVAL_REPORT.md` for the full 2×2 (leak × tracking):
| protocol \ model | leaked `_2` | leak-free `_80` |
|---|---|---|
| matching-only (GT tracklets) | 99.0 | 76.7 |
| full pipeline (det+MOT) | 46.8 | 34.6 |
- Leak ≈ 12–22 IDF1; **detection+MOT fragmentation ≈ 42–52 IDF1 (the dominant bottleneck)**.

**S06 official baseline:** 3-model ensemble + zone matcher → **IDF1 0.8090** (leak-free; matches the
published top-1 ~0.8095). Submitted via 2022 Track 1; log in `~/workspace/thesis-docs/submissions/README.md`.

**Real-time profiling** (`REALTIME_PROFILE.md`): ReID is the #1 GPU cost; MOT post-processing and
cross-camera clustering are architecturally offline.

Caveats: graph ablation at prototype scale (113 train identities), SimGNN tested not MGMN; the leak
inflates absolute scores (relative comparisons remain valid). The leak-free retrain addresses this.
