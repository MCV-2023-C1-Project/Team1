from  utils import utils
from common.config import *

from pathlib import Path
from PIL.Image import Image
from typing import *

import kornia as k
import kornia.enhance as ke
import numpy as np


def get_grayscale_histogram_descriptor(img: np.ndarray, **kwargs):
    """
    Compute a grayscale histogram descriptor for the given image.

    Parameters:
        img (np.ndarray): Input grayscale image.
        **kwargs: Additional keyword arguments for ke.image_histogram2d.

    Returns:
        np.ndarray: Grayscale histogram descriptor feature.
    """
    feature, pdf =   ke.image_histogram2d(img, **kwargs)

    return feature

def get_normalized_rg_histogram_descriptor(img: np.ndarray, **kwargs):
    """
    Compute a normalized red-green (RG) histogram descriptor for the given image.

    Parameters:
        img (np.ndarray): Input image in RGB format.
        **kwargs: Additional keyword arguments for ke.image_histogram2d.

    Returns:
        np.ndarray: Concatenated normalized red and green histogram features.
    """

    R_norm = (img[0,:,:]/img.sum(axis=0))
    G_norm = (img[1,:,:]/img.sum(axis=0))

    hist_r, _ = np.histogram(R_norm, range=[0.0, 1.0])
    hist_g, _ = np.histogram(G_norm, range=[0.0, 1.0])
    R_feature = hist_r / np.sum(hist_r)
    G_feature =  hist_g / np.sum(hist_g)

    return np.concatenate((R_feature, G_feature), axis=-1)

def get_cummulative_histogram_descriptor(img: np.ndarray,channels:list=[1,2], **kwargs):
    """
    Compute a cumulative histogram descriptor for the given image.

    Parameters:
        img (np.ndarray): Input image.
        channels (list): List of channels to compute the cumulative histogram for. Default is [1, 2].
        **kwargs: Additional keyword arguments for np.histogram.

    Returns:
        np.ndarray: Cumulative histogram descriptor feature.
    """
    feature = np.array([])
    for c in channels:
        hist, pdf = np.histogram(img[:,:,c], **kwargs)
        cumulative = np.cumsum(hist)
        feature = np.concatenate((feature, cumulative), axis=-1)

    return feature


def get_piramidal_histogram_descriptor(img: np.array, steps:int=5, **kwargs):
    histogram_descriptor = np.array([])
    for s in range(steps):
        tiles = 2**s
        feature_s = get_multi_tile_histogram_descriptor(img=img, tiles=tiles, **kwargs)
        histogram_descriptor = np.concatenate((histogram_descriptor, feature_s), axis=-1)

    return histogram_descriptor
def get_multi_tile_histogram_descriptor(img: np.array, tiles:int=10, **kwargs):
    """
    Compute a multi-tile histogram descriptor for the given image.

    Parameters:
        img (np.array): Input image.
        tiles (int): Number of tiles to divide the image into. Default is 10.
        channel (int): Channel to compute the histogram for. Default is 1.
        **kwargs: Additional keyword arguments for np.histogram.

    Returns:
        list: List of histograms for each tile.
    """

    h,w, channels = img.shape
    feature = np.array([])
    k_size_i = img.shape[0]//tiles
    k_size_j = img.shape[1]//tiles
    for i in range(0, h - (h%tiles), k_size_i):
        for j in range(0, w - (w%tiles), k_size_j):
            hist, _ = np.histogram(img[i:i+k_size_i, j:j+k_size_j,:[1,2]], **kwargs)
            feature = np.concatenate((feature,(hist/np.sum(hist))), axis=-1)

    return feature

def get_channel_multi_tile_histogram_descriptor(img: np.array, tiles:int=10, channel:int=1,  **kwargs):
    """
    Compute a multi-tile histogram descriptor for the given image.

    Parameters:
        img (np.array): Input image.
        tiles (int): Number of tiles to divide the image into. Default is 10.
        channel (int): Channel to compute the histogram for. Default is 1.
        **kwargs: Additional keyword arguments for np.histogram.

    Returns:
        list: List of histograms for each tile.
    """

    h,w, channels = img.shape
    feature = np.array([])
    k_size_i = img.shape[0]//tiles
    k_size_j = img.shape[1]//tiles
    for i in range(0, h - (h%tiles), k_size_i):
        for j in range(0, w - (w%tiles), k_size_j):
            hist, _ = np.histogram(img[i:i+k_size_i, j:j+k_size_j, channel], **kwargs)
            feature = np.concatenate((feature,(hist/np.sum(hist))), axis=-1)

    return feature

def get_histogram_descriptor(img_array: np.array, **kwargs):
    """
    Compute a histogram descriptor for the given image array.

    Parameters:
        img_array (np.array): Input image array.
        **kwargs: Additional keyword arguments for ke.image_histogram2d.

    Returns:
        np.ndarray: Histogram and probability density function (pdf).
    """
    im = utils.image2tensor(img_array)
    histogram, pdf = ke.image_histogram2d(im, **kwargs)

    return histogram, pdf

