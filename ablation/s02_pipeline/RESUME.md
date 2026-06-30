# S02 Full-Pipeline Eval — Resume Guide

Goal: run the **full MTMC pipeline on scene S02** (cameras c006–c009) using the
**leak-free retrained ReID model** (`reid/reid_model/resnet101_ibn_a_80.pth`, trained with
S02 held out), with **appearance-only cross-camera matching** (S06 zone masks/rules are NOT
ported), then score IDF1 against `datasets/mtmc_eval/gt_S02.txt` with the official `eval.py`.

Env: `conda activate aic21-mtmc`. Single RTX 4070 SUPER (12 GB).

## Why appearance-only (no zones)
The zone-based `trajectory_fusion → sub_cluster → gen_res` chain is hardcoded for S06
(cameras 41–46, hand-drawn `zone/c04*.png`, per-camera topology rules in `utils/zone_intra.py`
and `utils/filter.py`'s `CAM_DIST`). S02 has none of these. So the genuine stages
(detection → ReID → MOT) are run as-is, and the cross-camera step is replaced by a clean
appearance-only clusterer (`cluster_and_eval_s02.py`) fed the predicted MOT tracklets.

## Files (all in this dir unless noted)
- `config/aic_s02.yml` — detection + MOT config. DATA_DIR=`datasets/detect_merge_s02/`,
  DET_SOURCE_DIR=`datasets/detection/images/test/S02/`, ROI_DIR=validation S02,
  CID_BIAS_DIR=`datasets/cam_timestamp/`, MCMT_OUTPUT_TXT=`track_s02.txt`.
- `config/aic_reid_s02.yml` — ReID extract. REID_MODEL=`reid_model/resnet101_ibn_a_80.pth`,
  DET_IMG_DIR=`datasets/detect_merge_s02/`, DATA_DIR=`datasets/detect_reid1_s02/`.
- `gen_images_s02.py` — extract S02 frames (ROI-masked) → detection/images/test/S02/<cam>/img1/.
- `gen_det_s02.sh` — YOLOv5x detection for c006–c009 (run from `detector/yolov5/`).
- `merge_reid_feat_s02.py` — single-model merge: detect_reid1_s02 → detect_merge_s02.
- `run_aic_s02.sh` — MOT (FairMOT) for c006–c009, serial (run from `tracker/MOTBaseline/`).
- `cluster_and_eval_s02.py` — appearance-only cross-cam cluster + IDF1 via eval.py.

## Progress (DONE — on disk, do NOT redo)
- [x] Configs + S02 driver scripts written.
- [x] Frames: `datasets/detection/images/test/S02/{c006,c007,c008,c009}/img1/` (2110/1965/1924/2110).
- [x] Detection: `datasets/detect_merge_s02/<cam>/` — 130,490 crops + `<cam>_dets.pkl`
      (c006 61035, c007 5646, c008 25842, c009 37967).
- [x] ReID features (new model) extracted + merged: `datasets/detect_merge_s02/<cam>/<cam>_dets_feat.pkl`
      (c006 484M, c007 45M, c008 205M, c009 302M).

## TODO (resume here)
### Step 5 — MOT (single-camera tracking)  [was paused mid-c006; no output yet → restarts clean]
```bash
cd /home/likef/workspace/AIC21-MTMC/tracker/MOTBaseline
conda run -n aic21-mtmc bash ../../ablation/s02_pipeline/run_aic_s02.sh aic_s02.yml \
  > ../../ablation/s02_pipeline/logs/04_mot.log 2>&1
```
~10–15 min total (CPU-bound, serial; c006 largest). Produces per camera in
`datasets/detect_merge_s02/<cam>/`: `<cam>_mot_feat_raw.pkl` (fair_app) then
`<cam>_mot_feat.pkl` (post_processing — the file Step 6 consumes).
Verify: `ls datasets/detect_merge_s02/c00*/c00*_mot_feat.pkl` → 4 files.

### Step 6+7 — Cross-camera cluster + IDF1
```bash
cd /home/likef/workspace/AIC21-MTMC/ablation/s02_pipeline
conda run -n aic21-mtmc python cluster_and_eval_s02.py --thresholds 0.3 0.4 0.5 0.6 0.7
```
Prints `IDF1 / IDP / IDR` per cosine threshold. Writes preds to
`datasets/mtmc_eval/preds_s02/pred_cosine_thr<t>.txt`. Pick the best threshold.

## RESULT (2026-06-27 — pipeline complete)
First leak-free end-to-end MTMC on S02. MOT produced 2697 single-cam tracklets (vs 145 true
GT IDs → heavy fragmentation). Best appearance-only cross-cam IDF1:

| thr | clusters | IDF1 | IDP | IDR |
|-----|----------|------|-----|-----|
| 0.55 | 605 | 32.82 | 24.70 | 48.90 |
| 0.58 | 708 | 34.22 | 26.15 | 49.49 |
| **0.60** | **779** | **34.61** | **26.79** | **48.87** |
| 0.62 | 864 | 32.62 | 25.80 | 44.33 |
| 0.65 | 975 | 33.50 | 27.44 | 42.99 |

**Best: IDF1 34.61 @ thr 0.60.** Dominant error source is MOT fragmentation (2697 tracklets
for 145 IDs), not the ReID — to isolate ReID, run the GT-tracklet eval with the new model
(`eval_mtmc_validation.py`, see RETRAIN_PLAN Step 5).

## COMPARISON: leak-free (resnet101_ibn_a_80) vs leaked original (resnet101_ibn_a_2)
Same detection crops reused; only ReID→merge→MOT→cluster redone per model. Leaked-model
artifacts isolated in `*_s02_m2` dirs (`detect_reid1_s02_m2`, `detect_merge_s02_m2`,
`mtmc_eval/preds_s02_m2`); configs `aic_reid_s02_m2.yml` + `aic_s02_m2.yml`.

| model | MOT tracklets | best IDF1 | @thr | IDP | IDR |
|-------|---------------|-----------|------|-----|-----|
| leak-free `_80` (S02 held out) | 2697 | **34.61** | 0.60 | 26.79 | 48.87 |
| leaked `_2` (S02 in train data) | 2715 | **46.83** | 0.70 | 40.58 | 55.37 |

Leaked-model full sweep: thr 0.5→35.1, 0.6→41.2, 0.65→44.1, **0.70→46.8 (peak)**,
0.72→46.2, 0.75→43.9, 0.80→43.5, 0.85→25.0.

**Takeaways:**
- Tracklet counts are ~identical (2697 vs 2715) — MOT fragmentation is driven by detection +
  tracker, NOT the ReID model. So the IDF1 gap is purely cross-camera matching quality.
- The leak costs **~12 IDF1** (46.8 → 34.6), i.e. the leaked model's apparent advantage is
  largely memorized S02 identities. 34.6 is the honest figure.
- The leaked model peaks at a HIGHER threshold (0.70 vs 0.60): its same-vehicle cross-camera
  cosine sims are inflated, so it needs a stricter cut to avoid over-merging.

## 2×2 DECOMPOSITION: leak (model) × tracking (protocol)  — COMPLETE
GT-tracklet matching-only = `eval_mtmc_validation.py` (450 S02 GT tracklets, 4 cams, IDs
stripped, cosine). Leak-free cell built via `prepare_tracklet_graphs.py --reid-weights
resnet101_ibn_a_80.pth --out graphs_val_leakfree.pkl` then eval `--graphs-val` it.

| protocol \ model | leaked `_2` | leak-free `_80` |
|---|---|---|
| **GT tracklets (matching only)** | 99.03 (thr 0.7) | **76.66 (thr 0.63)** |
| **Full pipeline (det+MOT)** | 46.83 (thr 0.7) | 34.61 (thr 0.6) |

Decomposing the two error sources:
- **Leak effect** (same protocol, swap model): matching-only 99.0→76.7 = **~22 IDF1**;
  full pipeline 46.8→34.6 = **~12 IDF1**.
- **Detection+MOT effect** (same model, GT→real tracklets): leak-free 76.7→34.6 = **~42 IDF1**;
  leaked 99.0→46.8 = **~52 IDF1**.

**Key conclusion:** with the honest leak-free model, cross-camera *matching* is genuinely
solid (**76.7**). The full-pipeline collapse to 34.6 is overwhelmingly **tracker
fragmentation** (2697 tracklets for 145 vehicles), NOT ReID quality. The data leak inflated
the matching-only number by ~22 pts — the old "99" was both leaked and an idealized upper bound.

## Interpreting the result
This is the **first leak-free, end-to-end MTMC IDF1 on S02**. Compare against the leaked
GT-tracklet numbers (cosine ~99 / SimGNN ~63 from `ablation/simgnn/eval_mtmc_validation.py`).
Expect lower than 99: this number includes detection + MOT errors AND uses a ReID model that
never saw S02, so it is the honest figure. For an apples-to-apples ReID-only comparison,
also run `eval_mtmc_validation.py --graphs-val <new graphs from the leak-free model>` (GT
tracklets) — see `AICITY2021_Track2_DMT/RETRAIN_PLAN.md` Step 5.

## Reversibility / cleanup
All S02 artifacts are isolated in `*_s02` dirs (`detect_merge_s02`, `detect_reid1_s02`,
`detection/images/test/S02`, `mtmc_eval/preds_s02`). Removing them does not touch the S06 run.
