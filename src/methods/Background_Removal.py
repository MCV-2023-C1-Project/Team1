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




def iterative_searching_mask(img:np.ndarray, threshold:int=15, percent_img:float=0.3):
    height, width = img.shape[:2]
    half_width = int(width / 2)
    half_height = int(height / 2)
    mask = np.zeros((height, width), dtype=np.uint8)

    top_y = int(height * percent_img)
    bottom_y = int(height * (1 - percent_img))
    top_x = int(width * percent_img)
    bottom_x = int(width * (1 - percent_img))

    # Access pixel values at the specified locations
    left = img[half_height, 0, :].astype(np.int32)
    top = img[0, half_width, :].astype(np.int32)
    right = img[half_height, width - 1, :].astype(np.int32)
    bottom = img[height - 1, half_width, :].astype(np.int32)
    start_x, start_y, end_x, end_y = 0, 0, 0, 0

    # Top to bottom
    for y in range(1, top_y):
        dif = np.mean(np.abs(top - img[y, half_width, :]))
        if dif > threshold:
            break
        top = img[y, half_width, :].astype(np.int32)
        start_y = y

    # Bottom to top
    for y in range(height - 2, bottom_y, -1):
        dif = np.mean(np.abs(bottom - img[y, half_width, :]))
        if dif > threshold:
            break
        bottom = img[y, half_width, :].astype(np.int32)
        end_y = y

    # Left to right
    for x in range(1, top_x):
        dif = np.mean(np.abs(left - img[half_height, x, :]))
        if dif > threshold:
            break
        left = img[half_height, x, :].astype(np.int32)
        start_x = x

    # Right to left
    for x in range(width - 2, bottom_x, -1):
        dif = np.mean(np.abs(right - img[half_height, x, :]))
        if dif > threshold:
            break
        right = img[half_height, x, :].astype(np.int32)
        end_x = x

    # Create mask
    mask[start_y:end_y, start_x:end_x] = 255

    # full-black masks make errors, so here is the temporary solution
    # TODO: come up with something better
    if mask.sum() == 0:
        mask = np.full_like(mask, 255, np.uint8)

    return mask

"""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        mask = self.get_mask(img)
        rows, cols = np.where(mask == 255)
        output = img[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1]
        return output
"""

