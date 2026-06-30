#!/bin/bash
# Full 3-model-ensemble S06 pipeline for the AIC Track1-2022 / Track3-2021 submission.
# Detection + reid1 already on disk -> skip. Runs reid2, reid3, merge(3), MOT, cross-cam, gen_res.
# Launch with:  conda run -n aic21-mtmc bash run_s06_ensemble.sh   (env active inside)
set -e
REPO=/home/likef/workspace/AIC21-MTMC
cd $REPO

echo "########## [1/6] ReID extract: resnext101_ibn_a (aic_reid2) ##########"
cd $REPO/reid && python extract_image_feat.py aic_reid2.yml

echo "########## [2/6] ReID extract: resnet101_ibn_a_3 (aic_reid3) ##########"
cd $REPO/reid && python extract_image_feat.py aic_reid3.yml

echo "########## [3/6] Merge 3-model features -> detect_merge ##########"
cd $REPO/reid && python merge_reid_feat.py aic_all.yml

echo "########## [4/6] MOT (serial, c041-c046) ##########"
cd $REPO/tracker/MOTBaseline && bash $REPO/ablation/s06_ensemble/run_aic_s06_serial.sh aic_all.yml

echo "########## [5/6] Cross-camera: trajectory_fusion + sub_cluster ##########"
cd $REPO/reid/reid-matching/tools
python trajectory_fusion.py aic_all.yml
python sub_cluster.py aic_all.yml

echo "########## [6/6] gen_res -> track3.txt ##########"
cd $REPO/reid/reid-matching/tools && python gen_res.py aic_all.yml

echo "########## ENSEMBLE S06 PIPELINE DONE ##########"
wc -l $REPO/reid/reid-matching/tools/track3.txt
