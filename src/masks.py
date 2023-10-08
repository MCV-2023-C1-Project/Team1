from PIL import Image
import numpy as np
import kornia as K
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

    hist, pdf = Colors_Descriptors.get_multi_tile_histogram_descriptor(slice_list, **kwargs)
    return hist, pdf


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
    hist_list, pdf_list = extract_edge_histograms(img, n_bins = n_bins, return_pdf = True, min = min, max = max)

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


def get_mask_dict(folder: str) -> Dict[str, np.ndarray]:
    results_dict = {}

    img_list = utils.read_bbdd(folder)
    for img_path in img_list:
        og_img = Image.open(img_path)
        prediction = find_mask(utils.convert2rgchromaticity(K.tensor_to_image(K.color.rgb_to_linear_rgb(utils.image2tensor(og_img)))))
        results_dict[img_path.name] = prediction
    return results_dict