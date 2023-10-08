from PIL import Image
import numpy as np
from typing import Dict, Type, List, Tuple

from utils import utils
from methods import Colors_Descriptors


def extract_edge_histograms(img: np.ndarray, **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    height, width, _ = img.shape
    slice_list = []

    slice_list.append(img[2:3, :, :])
    slice_list.append(img[:, 2:3, :])
    slice_list.append(img[height-3:height-2, :, :])
    slice_list.append(img[:, width-3:width-2, :])
    total_hist = []
    for i in slice_list:
        for channel in range(i.shape[2]):
            hist = Colors_Descriptors.get_multi_tile_histogram_descriptor(i, tiles=1, channel=channel, **kwargs)
            total_hist.append(hist)

    return total_hist


def get_color_mask(img: np.ndarray, color_val: Tuple[float, float, float], tolerance: float) -> np.ndarray:
    channel_masks = np.empty(img.shape)
    for i in range(img.shape[2]):
        channel_masks[:, :, i] = (img[: , :, i] < color_val[i] + tolerance) & (img[: , :, i] > color_val[i] - tolerance)
    return 1 - np.all(channel_masks, 2).astype(np.int32)


def find_mask(img: np.ndarray) -> np.ndarray:
    # img should be in rg chromaticity
    n_bins = 32
    min = 0.
    max = 1.
    bin_width = (max - min)/n_bins
    hist_list = extract_edge_histograms(img, bins = n_bins, density = True, range=[min, max])

    hist_sum_r = np.zeros(hist_list[0].shape)
    hist_sum_g = np.zeros(hist_list[0].shape)
    hist_sum_b = np.zeros(hist_list[0].shape)
    for i in range(0, len(hist_list)//3, 3):
        hist_sum_r += hist_list[i]
        hist_sum_g += hist_list[i + 1]
        hist_sum_b += hist_list[i + 2]
    r_val = np.argmax(hist_sum_r) * bin_width
    g_val = np.argmax(hist_sum_g) * bin_width
    b_val = np.argmax(hist_sum_b) * bin_width
    mask = get_color_mask(img, (r_val, g_val, b_val), bin_width*2)
    return mask


