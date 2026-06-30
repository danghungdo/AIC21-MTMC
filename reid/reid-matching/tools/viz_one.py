"""Visualize MCMT result (track3.txt) for a SINGLE camera.
Usage: python viz_one.py <config.yml> <cam_int>   e.g. python viz_one.py aic_all.yml 41
"""
import os
import sys
sys.path.append('../../../')
from config import cfg
import argparse
from viz_mcmt import viz_mcmt

cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
cfg.freeze()
target_cam = int(sys.argv[2])

args = argparse.Namespace()
args.data_path = cfg.CHALLENGE_DATA_DIR
args.output_path = '../../../exp/viz'
args.mcmt_path = cfg.MCMT_OUTPUT_TXT

cam_track = {}
with open(args.mcmt_path) as f:
    for line in f:
        c, cid, fr, x, y, w, h, _, _ = tuple(int(float(s)) for s in line.split(' '))
        if c != target_cam:
            continue
        cam_track.setdefault(fr, []).append((cid, x, y, w, h))

print(f'cam c0{target_cam}: {len(cam_track)} annotated frames')
viz_mcmt(args, target_cam, cam_track)
print('done ->', os.path.join(args.output_path, 'mcmt', 'S06', f'c0{target_cam}.mp4'))
