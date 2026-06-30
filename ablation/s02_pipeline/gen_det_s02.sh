#!/bin/bash
# YOLOv5 detection for S02 cameras. Run from detector/yolov5/.
# Serial (single 12GB GPU). Writes crops+pkl to DATA_DIR/<cam>/ (detect_merge_s02).
seqs=(c006 c007 c008 c009)
for seq in ${seqs[@]}
do
    CUDA_VISIBLE_DEVICES=0 python detect2img.py --name ${seq} --weights yolov5x.pt \
        --conf 0.1 --agnostic --save-txt --save-conf --img-size 1280 --classes 2 5 7 \
        --cfg_file aic_s02.yml
done
