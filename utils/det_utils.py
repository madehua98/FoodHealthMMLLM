import numpy as np
import cv2


def to_color_map(diff):
    diff = diff / diff.max() * 255.0
    diff = diff.astype(np.uint8)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)


def compute_bbox_from_mask(mask):
    vs, us = np.nonzero(mask)
    x0, y0, x1, y1 = us.min().item(), vs.min().item(), us.max().item() + 1, vs.max().item() + 1
    return [x0, y0, x1, y1]
