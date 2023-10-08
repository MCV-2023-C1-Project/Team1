from utils import utils
from methods import Colors_Descriptors as CD
from methods.Background_Removal import *

from tqdm import tqdm
from PIL import Image

from typing import *

import kornia as K
import numpy as np

import os


def generate_K_response(descriptors_bdr:Dict[str, np.ndarray],
                      descriptors_queries:Dict[str, np.ndarray],
                      sim_func: Callable,
                      k:int = 1):
    """
    Generate K responses for each query descriptor using a similarity function.

    Parameters:
        descriptors_bdr (Dict[str, np.ndarray]): Dictionary of descriptors for the database images.
        descriptors_queries (Dict[str, np.ndarray]): Dictionary of descriptors for the query images.
        sim_func (Callable): Similarity function to compute the similarity between descriptors.
        k (int): Number of responses to generate for each query. Default is 1.

    Returns:
        List[List[int]]: List of K responses for each query descriptor.
    """

    final_responses = [[] for _ in range(len(descriptors_queries))]

    for idx, descriptor in enumerate((descriptors_queries.values())):
        scoring_list = (utils.retriev_image(descriptors_bdr, descriptor, distance=sim_func))
        final_responses[idx] += [ind[1] for ind in scoring_list[:k]]

    return final_responses

def generate_mask_dict(imgs:List[np.ndarray]) -> Dict[str, np.ndarray]:
    results_dict = {}

    for img_path in imgs:
        og_img = Image.open(img_path)
        prediction = find_mask(utils.convert2rgchromaticity(K.tensor_to_image(K.color.rgb_to_linear_rgb(utils.image2tensor(og_img)))))
        results_dict[img_path.name] = prediction
    return results_dict
def generate_grayscale_histogram_descriptors(imgs:List[np.ndarray], **kwargs):
    """
    Generate grayscale histogram descriptors for a list of images.

    Parameters:
        imgs (List[np.ndarray]): List of input images.
        **kwargs: Additional keyword arguments for CD.get_grayscale_histogram_descriptor.

    Returns:
        Dict[int, np.ndarray]: Dictionary of grayscale histogram descriptors for each image.
    """
    descriptors = {}
    for idx, img in tqdm(enumerate(imgs)):
        descriptors[idx] = CD.get_grayscale_histogram_descriptor(**kwargs)

    return descriptors


def generate_normalized_rg_histogram_descriptors(imgs:List[np.ndarray], **kwargs):
    """
    Generate normalized red-green (RG) histogram descriptors for a list of images.

    Parameters:
        imgs (List[np.ndarray]): List of input images.
        **kwargs: Additional keyword arguments for CD.get_normalized_rg_histogram_descriptor.

    Returns:
        Dict[int, np.ndarray]: Dictionary of normalized RG histogram descriptors for each image.
    """
    descriptors = {}
    for idx, img in tqdm(enumerate(imgs)):
        feature = CD.get_normalized_rg_histogram_descriptor(img.squeeze(), **kwargs)
        descriptors[idx] = feature
    return descriptors


def generate_cummulative_histogram_descriptors(imgs: np.ndarray, channels:list = [1,2], **kwargs) -> Dict[int, np.ndarray]:
    """
    Generate cumulative histogram descriptors for a list of images.

    Parameters:
        imgs (np.ndarray): Array of input images.
        channels (list): List of channels to compute the cumulative histogram for. Default is [1, 2].
        **kwargs: Additional keyword arguments for CD.get_cummulative_histogram_descriptor.

    Returns:
        Dict[int, np.ndarray]: Dictionary of cumulative histogram descriptors for each image.
    """
    descriptors = {}
    for idx, img in tqdm(enumerate(imgs)):
        feature = CD.get_cummulative_histogram_descriptor(img, channels, **kwargs)
        descriptors[idx] = feature

    return descriptors

def generate_multi_tile_histogram_descriptors(imgs:List[np.ndarray], tiles:int=10, channels:list=[1,2], **kwargs):
    """
    Generate multi-tile histogram descriptors for a list of images.

    Parameters:
        imgs (List[np.ndarray]): List of input images.
        tiles (int): Number of tiles to divide the image into. Default is 10.
        channel (int): Channel to compute the histogram for. Default is 1.
        **kwargs: Additional keyword arguments for CD.get_multi_tile_histogram_descriptor.

    Returns:
        Dict[int, np.ndarray]: Dictionary of multi-tile histogram descriptors for each image.
    """
    descriptors = {}
    for idx, img in tqdm(enumerate(imgs)):
        descriptors[idx] = np.array([])
        for channel in channels:
            feature = CD.get_multi_tile_histogram_descriptor(img, tiles, channel, **kwargs)
            descriptors[idx] = np.concatenate((descriptors[idx], feature.squeeze()), axis=-1)

    return descriptors

def get_h2_descriptors(imgs:List[np.ndarray], **kwargs) -> Dict[int, np.ndarray]:
    """
    Generate histogram descriptors for a list of images.

    Parameters:
        imgs (List[np.ndarray]): List of input images.
        **kwargs: Additional keyword arguments for CD.get_histogram_descriptor.

    Returns:
        Dict[int, np.ndarray]: Dictionary of histogram descriptors for each image.
    """
    descriptors = {}
    for idx, img in tqdm(enumerate(imgs)):
        hisogram, pdf = CD.get_histogram_descriptor(img[:, :, 1:], **kwargs)

        descriptors[idx] = pdf

    return descriptors

def get_mth1_descriptors(imgs:List[np.ndarray], colorspace:Callable, tiling:int=16,  **kwargs) -> Dict[int, np.ndarray]:
    """
    Compute MTH1 descriptors for a list of images.

    Parameters:
        imgs (List[np.ndarray]): A list of NumPy arrays representing images.
        tiling (int, optional): Tiling parameter. Defaults to 16.
        **kwargs: Additional keyword arguments for histogram calculation.

    Returns:
        Dict[int, np.ndarray]: A dictionary where keys are image indices and values are MTH1 descriptors.
    """
    descriptors =  {}
    for idx, img in tqdm(enumerate(imgs)):
        name = os.path.basename(img)
        img_tiles = utils.get_slices_from_image(img, tiling)
        dif_colors = utils.transform_tiles_colorspace(img_tiles, colorspace=colorspace)
        c1, c2, c3 = utils.split_nd_tiles_colorspace(dif_colors)

        histogram_c2, pdf_c2 = CD.get_multi_tile_histogram_descriptor(tiles=c2,**kwargs)
        histogram_c3, pdf_c3 = CD.get_multi_tile_histogram_descriptor(tiles=c3, **kwargs)

        c2_descriptor = np.zeros(kwargs["n_bins"])
        c3_descriptor = np.zeros(kwargs["n_bins"])

        for i, j in zip(pdf_c2, pdf_c3):
            c2_descriptor += i
            c3_descriptor += j

        img_descriptor = np.concatenate((c2_descriptor, c3_descriptor), axis=-1)

        descriptors[name] = img_descriptor

    return descriptors

