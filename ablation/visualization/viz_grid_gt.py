"""Ground-truth 2x2 grid clip for S02 (cameras c006-c009).

Like reid/reid-matching/tools/viz_grid.py, but draws the GROUND-TRUTH boxes from
gt.txt (global vehicle IDs) instead of pipeline predictions. Same color = same
vehicle across cameras, so you can see the true cross-camera identities.

S02 cameras are NOT frame-synced (cam_timestamp offsets 0/0.06/0.42/0.66 s), so
each camera is shifted by round(offset*fps) frames to align by wall-clock.

Usage (from ablation/visualization):
  python viz_grid_gt.py
"""
import os
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
S02_DIR = os.path.join(REPO, 'datasets', 'siam_raw', 'validation', 'S02')
TS_FILE = os.path.join(REPO, 'datasets', 'cam_timestamp', 'S02.txt')
OUT = os.path.join(REPO, 'datasets', 'siam', 'grid_S02_gt.mp4')

CAMS = [6, 7, 8, 9]          # row-major: c006 c007 / c008 c009
COLS, ROWS = 2, 2
TILE_W, TILE_H = 640, 360
FPS = 10

COLORS = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0),
          (210, 105, 30), (220, 20, 60), (192, 192, 192), (255, 228, 196), (50, 205, 50),
          (139, 0, 139), (100, 149, 237), (138, 43, 226), (238, 130, 238), (255, 0, 255),
          (0, 100, 0), (127, 255, 0), (0, 0, 205), (255, 140, 0), (199, 21, 133),
          (124, 252, 0), (147, 112, 219), (106, 90, 205), (65, 105, 225), (255, 20, 147),
          (186, 85, 211), (148, 0, 211), (255, 99, 71), (0, 191, 255), (60, 179, 113)]


def draw(img, boxes):
    for gid, x, y, w, h in boxes:
        c = COLORS[gid % len(COLORS)]
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 3)
        cv2.rectangle(img, (x, y - 22), (x + 11 * len(str(gid)) + 6, y), c, -1)
        cv2.putText(img, str(gid), (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def parse_gt(cam):
    """local 1-based frame -> list of (gid, x, y, w, h)."""
    per_frame = defaultdict(list)
    gt = os.path.join(S02_DIR, f'c{cam:03d}', 'gt', 'gt.txt')
    for line in open(gt):
        p = line.strip().split(',')
        if len(p) < 6:
            continue
        fr, gid = int(p[0]), int(p[1])
        x, y, w, h = (int(float(p[2])), int(float(p[3])), int(float(p[4])), int(float(p[5])))
        per_frame[fr].append((gid, x, y, w, h))
    return per_frame


def main():
    biases = {}
    for line in open(TS_FILE):
        c, b = line.split()
        biases[int(c[1:])] = float(b)
    offset = {c: round(biases.get(c, 0.0) * FPS) for c in CAMS}   # frames each cam is "late"
    max_off = max(offset.values())
    lead = {c: max_off - offset[c] for c in CAMS}                 # frames to skip at start (wall-clock align)

    gt = {c: parse_gt(c) for c in CAMS}
    caps = {c: cv2.VideoCapture(os.path.join(S02_DIR, f'c{c:03d}', 'vdo.avi')) for c in CAMS}
    nframes = {c: int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) for c in CAMS}
    for c in CAMS:
        for _ in range(lead[c]):           # skip lead frames to align by wall-clock
            caps[c].read()
    print('offsets(frames):', offset, ' lead:', lead)
    total = min(nframes[c] - lead[c] for c in CAMS)
    print(f'grid frames: {total}  ({len(CAMS)} cams, {sum(len(v) for v in gt.values())} GT boxes)')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), float(FPS), (TILE_W * COLS, TILE_H * ROWS))
    local_idx = {c: lead[c] for c in CAMS}   # 0-based local frame already positioned at lead
    for g in range(total):
        tiles = []
        for c in CAMS:
            ok, im = caps[c].read()
            if not ok:
                im = np.zeros((TILE_H, TILE_W, 3), 'uint8')
            else:
                gt_frame = local_idx[c] + 1                     # gt.txt is 1-based
                im = draw(im, gt[c].get(gt_frame, []))
                im = cv2.resize(im, (TILE_W, TILE_H))
            local_idx[c] += 1
            cv2.putText(im, f'c{c:03d} f{local_idx[c]}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(im)
        rows = [cv2.hconcat(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
        writer.write(cv2.vconcat(rows))
        if g % 300 == 0:
            print(f'{g}/{total}')
    writer.release()
    for cap in caps.values():
        cap.release()
    print('done ->', OUT)


if __name__ == '__main__':
    main()
