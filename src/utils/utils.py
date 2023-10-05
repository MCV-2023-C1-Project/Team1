## from ....
from typing import *
from kornia.contrib import (
    CombineTensorPatches,
    ExtractTensorPatches,
    combine_tensor_patches,
    extract_tensor_patches,
)


from pathlib import  Path
from PIL import Image


import numpy as np
import kornia as K
import kornia.enhance as ke
import image_slicer as slicer


import imutils




def get_slices_from_image(path: str, ntiles:int) -> List[Path]:
    """
    Get image slices from the specified image.

    Parameters:
        path (str): The path to the input image.
        ntiles (int): The number of slices/tiles to generate.

    Returns:
        List[Path]: A list of Path objects representing the generated slices.
    """
    tiles = slicer.slice(path, ntiles, save=False)

    return tiles

def save_tiles(tiles: List[Path], savefile: Path, **kwargs) -> None:
    """
    Save image slices/tiles to the specified directory.

    Parameters:
        tiles (List[Path]): A list of Path objects representing image slices.
        savefile (Path): The directory where the slices will be saved.
        **kwargs: Additional keyword arguments for customization.
    """
    slicer.save_tiles(tiles, directory=savefile, **kwargs)


def read_bbdd(path: Type[str]) -> List[Type[Path]]:
    """
    Reads image files from the specified directory path and returns a list of image file names.

    Args:
        path (Type[str]): The directory path where the images are located.
        x (List[int, ...]): Unused argument (appears to be a placeholder).

    Returns:
        List[Path, ...]: A list of image file names (with the .jpg extension) in the specified directory.
    """

    p = Path(path)
    img_list = list(p.glob("*.jpg")) # lista [~/BBDD/bbdd_0000.jpg ...]

    return img_list


def image2tensor(img: Image) -> np.ndarray:

    """
    Convert an image to a tensor.

    Parameters:
        img (Image): The input image as an instance of the Image class.

    Returns:
        np.ndarray: The image represented as a NumPy array (tensor).
    """
    im = np.array(img)
    return K.image_to_tensor(im)
def convert2gray(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.

    Parameters:
        img (np.ndarray): The input RGB image as a NumPy array.

    Returns:
        np.ndarray: The grayscale image as a NumPy array.
    """
    return K.tensor_to_image(K.color.rgb_to_grayscale(img))

def convert2luv(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to LUV color space.

    Parameters:
        img (np.ndarray): The input RGB image as a NumPy array.

    Returns:
        np.ndarray: The image in LUV color space as a NumPy array.
    """
    return K.tensor_to_image(K.color.rgb_to_luv(img))

def convert2lab(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to LAB color space.

    Parameters:
        img (np.ndarray): The input RGB image as a NumPy array.

    Returns:
        np.ndarray: The image in LAB color space as a NumPy array.
    """
    return K.tensor_to_image(K.color.rgb_to_lab(img))

def apply_min_max_scaler(img:np.ndarray, **kwargs):
    """
    Apply min-max scaling to the input image.

    Parameters:
        img (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The scaled image as a NumPy array.
    """
    return ke.normalize_min_max(img)

def extract_tiles(img: np.ndarray, weight:int, height:int, window_size: int) -> List[np.ndarray]:

    h_pad = (window_size - (height % window_size)) // 2
    w_pad = (window_size - (weight % window_size)) // 2
    padding = (h_pad, w_pad)

    img_tiles = extract_tensor_patches(img, window_size=window_size, stride=w_pad, padding=padding)

    return img_tiles



def read_img(img_path: Path):
    """
    Opens an image file specified by the provided file path using the PIL library.

    Args:
        img_path (Path): The path to the image file.

    Returns:
        Image.Image: A PIL Image object representing the opened image.
    """
    return K.io.load_image(img_path, K.io.ImageLoadType.RGB32)
    #return Image.open(img_path)





