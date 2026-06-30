"""Combine all 6 S06 cameras into a single 2x3 grid clip, annotated with
global vehicle IDs from track3.txt (same ID/color = same vehicle across cams).

Usage: python viz_grid.py <config.yml>   e.g. python viz_grid.py aic_all.yml
Aligns cameras by frame index (not wall-clock; see cam_timestamp for offsets).
"""
import os
import sys
sys.path.append('../../../')
from config import cfg
import cv2
from viz_mcmt import draw_bboxes

CAMS = [41, 42, 43, 44, 45, 46]   # row-major: row0 = 41,42,43 ; row1 = 44,45,46
COLS, ROWS = 3, 2
TILE_W, TILE_H = 640, 360          # output = 1920 x 720

cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
cfg.freeze()
data_path = cfg.CHALLENGE_DATA_DIR
out_dir = '../../../exp/viz/mcmt/S06'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'grid_S06.mp4')

# ---- load track3.txt grouped by camera then frame ----
cam_tracks = {c: {} for c in CAMS}
with open(cfg.MCMT_OUTPUT_TXT) as f:
    for line in f:
        c, cid, fr, x, y, w, h, _, _ = tuple(int(float(s)) for s in line.split(' '))
        if c in cam_tracks:
            cam_tracks[c].setdefault(fr, []).append((cid, x, y, w, h))

# ---- open all camera videos ----
caps, nframes = {}, {}
for c in CAMS:
    vp = os.path.join(data_path, 'test', 'S06', f'c0{c}', 'vdo.avi')
    cap = cv2.VideoCapture(vp)
    caps[c] = cap
    nframes[c] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'c0{c}: {nframes[c]} frames, {len(cam_tracks[c])} annotated')

total = max(nframes.values())
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 25.0,
                         (TILE_W * COLS, TILE_H * ROWS))

def make_tile(c, fr_id):
    cap = caps[c]
    state, im = cap.read()
    if not state or im is None:
        return None  # camera exhausted
    if fr_id in cam_tracks[c]:
        bbox_xyxy, pids = [], []
        for pid, x, y, w, h in cam_tracks[c][fr_id]:
            bbox_xyxy.append([x, y, x + w, y + h])
            pids.append(pid)
        im = draw_bboxes(im, bbox_xyxy, pids)
    tile = cv2.resize(im, (TILE_W, TILE_H))
    cv2.putText(tile, f'c0{c} f{fr_id}', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return tile

black = None
for fr_id in range(1, total + 1):
    tiles = []
    for c in CAMS:
        t = make_tile(c, fr_id)
        if t is None:
            if black is None:
                import numpy as np
                black = np.zeros((TILE_H, TILE_W, 3), dtype='uint8')
            t = black
        tiles.append(t)
    rows = [cv2.hconcat(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
    grid = cv2.vconcat(rows)
    writer.write(grid)
    if fr_id % 200 == 0:
        print(f'{fr_id}/{total}')

writer.release()
for cap in caps.values():
    cap.release()
print('done ->', out_path)
