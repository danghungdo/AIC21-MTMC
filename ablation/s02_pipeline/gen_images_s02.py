"""Extract S02 (validation) frames into the pipeline's DET_SOURCE_DIR/<cam>/img1/ layout,
applying the same ROI ignore-region masking as detector/gen_images_aic.py.

S02 raw videos live under the validation tree (not test/), so this standalone script
reads them directly instead of going through gen_images_aic.py's hardcoded 'test' walk.
"""
import os
import cv2
from tqdm import tqdm

S02_ROOT = '/home/likef/workspace/city-flow-raw/cityflow-raw-data/validation-002/validation/S02'
DST = '/home/likef/workspace/AIC21-MTMC/datasets/detection/images/test/S02'
CAMS = ['c006', 'c007', 'c008', 'c009']


def draw_ignore_regions(img, region):
    return img * (region > 0)


def main():
    for cam in CAMS:
        vpath = os.path.join(S02_ROOT, cam, 'vdo.avi')
        roi = cv2.imread(os.path.join(S02_ROOT, cam, 'roi.jpg'))
        out_dir = os.path.join(DST, cam, 'img1')
        os.makedirs(out_dir, exist_ok=True)
        video = cv2.VideoCapture(vpath)
        n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        cur = 0
        with tqdm(total=n, desc=cam) as bar:
            while cur < n - 1:
                cur = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                ok, frame = video.read()
                if not ok:
                    break
                dst = os.path.join(out_dir, 'img{:06d}.jpg'.format(cur))
                if not os.path.isfile(dst):
                    cv2.imwrite(dst, draw_ignore_regions(frame, roi))
                bar.update(1)
        video.release()
        print(f'{cam}: wrote frames to {out_dir}')


if __name__ == '__main__':
    main()
