# -*- coding: utf-8 -*-
"""Два числа для REPORT.md, которых photolab.py не печатает (эталон есть только у кадров курса).

  python report_numbers.py ball  out/ck_mask.png                       → recall / precision против круга (190, 560), r = 58
  python report_numbers.py poster out/poster_course.png L4/poster-gt.png → доля совпавших пикселей после бинаризации
"""
import sys

import cv2
import numpy as np

from photolab import imread


def ball(mask_path, cx=190, cy=560, r=58):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    gt = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2           # эталонный диск мяча
    tp = (mask & gt).sum()
    print("recall %.1f %%, precision %.1f %%" % (100 * tp / gt.sum(), 100 * tp / max(mask.sum(), 1)))


def poster(front_path, gt_path):
    ink = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 128   # noqa: E731  бинаризация, как в тестах
    a, b = imread(front_path), imread(gt_path)
    if a.shape[:2] != b.shape[:2]:
        raise SystemExit("размеры не совпадают: %s и %s — запускайте rectify с --size %dx%d" % (a.shape[:2], b.shape[:2], b.shape[1], b.shape[0]))
    print("совпало с %s после бинаризации: %.1f %% пикселей" % (gt_path, 100 * (ink(a) == ink(b)).mean()))


if __name__ == "__main__":
    {"ball": ball, "poster": poster}[sys.argv[1]](*sys.argv[2:])
