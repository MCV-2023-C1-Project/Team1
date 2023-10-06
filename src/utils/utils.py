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

import pickle
import imutils
import os




def create_descriptor_database(filepath:str, filename:Optional[str] = None) -> None:
    db = {}
    path = os.path.join(filepath, filename) if filename is not None else filepath

    abs_path = os.path.dirname(path)
    os.makedirs(abs_path, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(db, f)

def get_descriptor_database(filepath:str, filename:Optional[str] = None) -> Dict[str, np.ndarray]:
    path = os.path.join(filepath, filename) if filename is not None else filepath

    with open(path, "rb") as f:
        load_descriptor = pickle.load(f)

    return load_descriptor

def merge_descriptor_database(descriptors:Dict[str, np.ndarray], filepath:str, overwrite:bool=False) -> bool:
    descriptors_db:Dict[str, np.ndarray] = utils.get_descriptor_database(filepath)
    flag = False
    for name_descriptor, information in descriptors.items():
        if (descriptors_db.get(name_descriptor, None) is None) or (overwrite is True):
            descriptors[name_descriptor] = information
            save_descriptor_bbdd(descriptors=descriptors, filepath=filepath)
            flag = True

    return flag

def save_descriptor_bbdd(descriptors: Dict[str, np.ndarray], filepath:str, filename:Optional[str]= None) -> None:
    if not os.path.exists(filepath):
        create_descriptor_database(filepath=filepath, filename=filename)

    path = os.path.join(filepath, filename) if filename is not None else filepath
    with open(path, "wb") as f:
        pickle.dump(descriptors, f)


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





