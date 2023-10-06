from utils import utils

from PIL.Image import Image

from typing import *

import numpy as np



def transform_tiles_colorspace(tiles: List[Image], colorspce:Callable)-> List[np.ndarray]:
    """
    Transform tiles' colorspace using the specified colorspace transformation function.

    Parameters:
        tiles (List[Image]): A list of images represented as instances of the Image class.
        colorspce (Callable): A callable representing a colorspace transformation function.

    Returns:
        List[np.ndarray]: A list of NumPy arrays representing the converted tiles.
    """
    converted_tiles = []
    for im in tiles:
        img = utils.image2tensor(im.image)
        img = utils.apply_min_max_scaler(img)
        img = colorspce(img)
        converted_tiles.append(img)

    return converted_tiles


def split_nd_tiles_colorspace(tiles: List[np.ndarray]) -> Tuple[np.ndarray]:
    """
    Split N-dimensional tiles into individual color channels.

    Parameters:
        tiles (List[np.ndarray]): A list of NumPy arrays representing N-dimensional color tiles.

    Returns:
        Tuple[np.ndarray]: A tuple containing N arrays representing individual color channels.
    """
    channels = [np.split(tile, tile.shape[-1], axis=-1) for tile in tiles]
    channel_arrays = tuple(np.array(channel) for channel in zip(*channels))
    return channel_arrays

# Example usage
# Assume 'tiles' is a list of N-dimensional color tiles
# channel_arrays = split_nd_tiles_colorspace(tiles)
# channel_arrays will contain the N arrays representing individual color channels

