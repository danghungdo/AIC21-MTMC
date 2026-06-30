"""Profile the S02 MTMC pipeline's GPU-bound stages for a real-time assessment.

Measures, on this machine's GPU:
  - video decode throughput (cv2)
  - YOLOv5x @1280 detection forward (FP16, as the pipeline runs it)
  - ReID resnet101-ibn @384 forward: FP32 vs FP16, single vs flip(2x)
  - ReID crop LOAD as a function of detection confidence threshold (from existing dets pkls)
Then compares to the 10 fps (CityFlow) real-time target.

Run from ablation/s02_pipeline/:  conda run -n aic21-mtmc python profile_pipeline.py
"""
import os
import pickle
import sys
import time
from glob import glob

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
S02_RAW = '/home/likef/workspace/city-flow-raw/cityflow-raw-data/validation-002/validation/S02'
DET_DIR = os.path.join(REPO, 'datasets', 'detect_merge_s02')
CAMS = ['c006', 'c007', 'c008', 'c009']
FPS = 10.0  # CityFlow native frame rate -> real-time target
dev = torch.device('cuda')


def hdr(t):
    print(f'\n{"="*68}\n{t}\n{"="*68}')


def sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------- crop load
def crop_load():
    hdr('1. ReID CROP LOAD vs detection confidence threshold')
    thrs = [0.1, 0.2, 0.3, 0.4, 0.5]
    totals = {t: 0 for t in thrs}
    for cam in CAMS:
        d = pickle.load(open(os.path.join(DET_DIR, cam, f'{cam}_dets.pkl'), 'rb'))
        confs = np.array([v['conf'] for v in d.values()])
        for t in thrs:
            totals[t] += int((confs >= t).sum())
    base = totals[0.1]
    print(f'{"conf>=":>8} {"crops":>10} {"vs conf0.1":>12}')
    for t in thrs:
        print(f'{t:>8} {totals[t]:>10} {totals[t]/base*100:>11.1f}%')
    return base


# ---------------------------------------------------------------- decode
def decode_bench():
    hdr('2. VIDEO DECODE (cv2, single thread)')
    cap = cv2.VideoCapture(os.path.join(S02_RAW, 'c007', 'vdo.avi'))
    n, t0 = 0, time.time()
    while n < 600:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    dt = time.time() - t0
    cap.release()
    print(f'decoded {n} frames in {dt:.2f}s -> {n/dt:.1f} fps  (target {FPS} fps/cam)')
    return n / dt


# ---------------------------------------------------------------- detection
def det_bench():
    hdr('3. DETECTION  YOLOv5x @1280, FP16 (forward only, no NMS/decode)')
    ydir = os.path.join(REPO, 'detector', 'yolov5')
    sys.path.insert(0, ydir)
    os.chdir(ydir)  # so relative weight/download paths resolve here
    from models.experimental import attempt_load
    w = os.path.join(ydir, 'yolov5x.pt')
    assert os.path.isfile(w), f'missing {w}'
    model = attempt_load(w, map_location=dev).half().eval()
    x = torch.zeros(1, 3, 1280, 1280, device=dev).half()
    with torch.no_grad():
        for _ in range(3):
            model(x)  # warmup
        sync(); t0 = time.time(); N = 30
        for _ in range(N):
            model(x)
        sync(); dt = time.time() - t0
    fps = N / dt
    print(f'{fps:.1f} img/s  ({dt/N*1000:.1f} ms/img)  -> {fps/FPS:.2f}x real-time for 1 cam, '
          f'{fps/(FPS*len(CAMS)):.2f}x for {len(CAMS)} cams')
    del model; torch.cuda.empty_cache()
    return fps


# ---------------------------------------------------------------- reid
def reid_bench(total_crops):
    hdr('4. ReID  resnet101-ibn @384 (forward only)')
    sys.path.insert(0, REPO)                       # -> config
    sys.path.insert(0, os.path.join(REPO, 'reid'))  # -> reid_inference
    os.chdir(os.path.join(REPO, 'reid'))            # so cfg.REID_MODEL relative path resolves
    from config import cfg
    cfg.merge_from_file(os.path.join(REPO, 'config', 'aic_reid_s02.yml'))
    cfg.freeze()
    from reid_inference.reid_model import build_reid_model
    model, _ = build_reid_model(cfg)
    model = model.to(dev).eval()
    B = 64

    def run(half, flip):
        m = model.half() if half else model.float()
        x = torch.zeros(B, 3, 384, 384, device=dev)
        x = x.half() if half else x.float()
        with torch.no_grad():
            for _ in range(3):
                m(x)
            sync(); t0 = time.time(); N = 20
            for _ in range(N):
                m(x)
                if flip:
                    m(torch.flip(x, [3]))
            sync(); dt = time.time() - t0
        return B * N / dt

    configs = [('FP32, flip(2x) [AS RUN]', False, True),
               ('FP32, single', False, False),
               ('FP16, single', True, False),
               ('FP16, no-flip [recommended]', True, False)]
    print(f'{"config":>32} {"crops/s":>10} {"time for "+str(total_crops)+" crops":>26}')
    rates = {}
    for name, half, flip in configs:
        r = run(half, flip)
        rates[name] = r
        print(f'{name:>32} {r:>10.0f} {total_crops/r:>22.1f} s')
    model.float()
    del model; torch.cuda.empty_cache()
    return rates


def main():
    print(f'GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}')
    base = crop_load()
    dec = decode_bench()
    det = det_bench()
    rates = reid_bench(base)

    hdr('SUMMARY vs real-time (10 fps/cam, 4 cams = 40 fps aggregate)')
    secs_footage = 2110 / FPS  # longest cam ~211s; 4 cams run concurrently in real life
    print(f'S02 footage: ~{secs_footage:.0f}s per camera (cams run concurrently)')
    print(f'Decode:    {dec:>7.1f} fps/cam  ({"OK" if dec>=FPS*len(CAMS) else "needs GPU decode / parallel"})')
    print(f'Detection: {det:>7.1f} img/s    ({det/(FPS*len(CAMS)):.2f}x the 4-cam aggregate need)')
    reid_as_run = rates['FP32, flip(2x) [AS RUN]']
    reid_best = rates['FP16, no-flip [recommended]']
    # crops/sec needed = avg crops/frame * 40 fps. avg crops/frame:
    cpf = base / (2110 + 1965 + 1924 + 2110)
    need = cpf * FPS * len(CAMS)
    print(f'ReID need: ~{cpf:.1f} crops/frame x {int(FPS*len(CAMS))} fps = {need:.0f} crops/s @conf0.1')
    print(f'  as-run  (FP32+flip): {reid_as_run:>7.0f} crops/s  -> {reid_as_run/need:.2f}x need')
    print(f'  best (FP16, no-flip):{reid_best:>7.0f} crops/s  -> {reid_best/need:.2f}x need')


if __name__ == '__main__':
    main()
