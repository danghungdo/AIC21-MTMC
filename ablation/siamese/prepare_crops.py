"""Step 3a data prep: extract ID-labeled vehicle crops from CityFlow gt.txt + vdo.avi.

For each camera, reads gt.txt (MOTChallenge: frame,id,left,top,w,h,...), decodes the
video, crops each annotated box, and saves to:
    <out>/<split>/<scene>_<id>/<scene>_<cam>_f<frame>_b<box>.jpg
Identity label = <scene>_<id>; AIC21 MTMC gt uses globally-consistent IDs within a
scene, so one label = the same vehicle across cameras (ideal for cross-camera ReID).

Per-identity stride subsampling keeps the crop count manageable for a prototype.

Usage:
    python prepare_crops.py --raw ../../datasets/siam_raw/train --split train --stride 5
    python prepare_crops.py --raw ../../datasets/siam_raw/validation --split val --stride 5
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_gt(gt_path):
    """frame(0-based) -> list of (id, left, top, w, h)."""
    per_frame = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            frame, oid = int(p[0]), int(p[1])
            l, t, w, h = (int(float(p[2])), int(float(p[3])),
                          int(float(p[4])), int(float(p[5])))
            per_frame[frame - 1].append((oid, l, t, w, h))  # gt is 1-based
    return per_frame


def process_camera(cam_dir, scene, cam, out_root, split, stride, min_size):
    gt_path, vid_path = cam_dir / 'gt' / 'gt.txt', cam_dir / 'vdo.avi'
    if not gt_path.exists() or not vid_path.exists():
        return 0
    per_frame = parse_gt(gt_path)
    cap = cv2.VideoCapture(str(vid_path))
    id_count = defaultdict(int)  # occurrences seen per id (for stride)
    saved, fidx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for bi, (oid, l, t, w, h) in enumerate(per_frame.get(fidx, [])):
            c = id_count[oid]
            id_count[oid] += 1
            if c % stride != 0:
                continue
            x1, y1 = max(0, l), max(0, t)
            x2, y2 = min(frame.shape[1], l + w), min(frame.shape[0], t + h)
            if x2 - x1 < min_size or y2 - y1 < min_size:
                continue
            label = f"{scene}_{oid:04d}"
            d = out_root / split / label
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{scene}_{cam}_f{fidx:06d}_b{bi}.jpg"),
                        frame[y1:y2, x1:x2])
            saved += 1
        fidx += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='dir holding <scene>/<cam>/{vdo.avi,gt/gt.txt}')
    ap.add_argument('--split', required=True, help='train | val')
    ap.add_argument('--out', default='../../datasets/siam/crops')
    ap.add_argument('--stride', type=int, default=5, help='keep every Nth occurrence per id')
    ap.add_argument('--min-size', type=int, default=32, help='skip crops smaller than this (px)')
    args = ap.parse_args()

    raw, out_root = Path(args.raw), Path(args.out)
    cam_dirs = sorted(raw.glob('S*/c*'))
    total = 0
    for cam_dir in tqdm(cam_dirs, desc=f'{args.split} cameras'):
        scene, cam = cam_dir.parent.name, cam_dir.name
        total += process_camera(cam_dir, scene, cam, out_root, args.split,
                                args.stride, args.min_size)
    n_ids = len(list((out_root / args.split).glob('*'))) if (out_root / args.split).exists() else 0
    print(f'[{args.split}] saved {total} crops across {n_ids} identities -> {out_root / args.split}')


if __name__ == '__main__':
    main()
