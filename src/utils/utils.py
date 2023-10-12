## from ....
from utils import utils
from typing import *



from pathlib import  Path
from PIL.Image import Image
from matplotlib import colors


import numpy as np
import kornia as K


import pickle
import os
import torch
import matplotlib.pyplot as plt




def RG_Chroma_plotter(red,green):
    p_color = [(r, g, 1-r-g) for r,g in
               zip(red.flatten(),green.flatten())]
    norm = colors.Normalize(vmin=0,vmax=1.)
    norm.autoscale(p_color)
    p_color = norm(p_color).tolist()
    fig = plt.figure(figsize=(10, 7), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(red.flatten(),
                green.flatten(),
                c = p_color, alpha = 0.40)
    ax.set_xlabel('Red Channel', fontsize = 20)
    ax.set_ylabel('Green Channel', fontsize = 20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.show()


def retriev_image(descriptors_bdr:Dict[str, np.ndarray], compare_descriptor: np.ndarray, distance:Callable) -> List[Tuple[float, str]]:
    """
    Retrieve images based on a similarity measure between descriptors.

    This function compares a given descriptor against a collection of descriptors and returns a sorted list of tuples
    representing the similarity scores and corresponding image names.

    Parameters:
    - descriptors_bdr (Dict[str, np.ndarray]): A dictionary mapping image names to their respective descriptors.
    - compare_descriptor (np.ndarray): The descriptor to compare against the collection of descriptors.
    - distance (Callable): A callable distance function to measure similarity between descriptors.

    Returns:
    - List[Tuple[float, str]]: A list of tuples, each containing a similarity score (between 0 and 1) and the image name.
      The list is sorted in descending order of similarity scores.
    """

    ## Have into account that we are assuming normlaized similarity where 1 is the maximun value
    results = []
    for idx, (name, descriptor) in enumerate(descriptors_bdr.items()):
        result =  distance(descriptor, compare_descriptor)
        results.append(tuple([result, idx]))

    final = sorted(results, reverse=True)


    return  final

def read_pickle(filepath:str) -> Any:
    """
    Read and deserialize a pickled object from the specified file.

    Parameters:
        filepath (str): The path to the file containing the pickled object.

    Returns:
        Any: The deserialized object from the pickled file.
    """
    with open(filepath, "rb") as file:
        load_file = pickle.load(file)



    return load_file

def write_pickle(information:Any,filepath:str):
    """
    Serialize and write an object to the specified file using pickle.

    Parameters:
        information (Any): The object to be serialized and written.
        filepath (str): The path to the file to write the pickled object.
    """

    abs_path = os.path.dirname(filepath)
    os.makedirs(abs_path, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(information, f)


def create_descriptor_database(filepath:str, filename:Optional[str] = None) -> None:
    """
    Create an empty descriptor database and write it to a specified file.

    Parameters:
        filepath (str): The directory where the database file will be stored.
        filename (str, optional): The name of the database file. Defaults to None.
    """
    db = {}
    path = os.path.join(filepath, filename) if filename is not None else filepath

    abs_path = os.path.dirname(path)
    os.makedirs(abs_path, exist_ok=True)

    write_pickle(db, path)

def get_descriptor_database(filepath:str, filename:Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Retrieve the descriptor database from a specified file.

    Parameters:
        filepath (str): The directory where the database file is stored.
        filename (str, optional): The name of the database file. Defaults to None.

    Returns:
        Tuple[Dict[str, np.ndarray], bool]: A tuple containing the loaded descriptor database and a flag indicating whether the database existed or not.
    """
    path = os.path.join(filepath, filename) if filename is not None else filepath
    check = False
    try:
        load_descriptor = read_pickle(path)
        check = True
    except:
        create_descriptor_database(filepath, filename)
        load_descriptor = {}

    return load_descriptor, check

def merge_descriptor_database(descriptors:Dict[str, np.ndarray], filepath:str, overwrite:bool=False) -> bool:
    """
    Merge a dictionary of descriptors into the descriptor database.

    Parameters:
        descriptors (Dict[str, np.ndarray]): A dictionary of descriptors to be merged into the database.
        filepath (str): The directory where the database file is stored.
        overwrite (bool, optional): Flag to indicate whether to overwrite existing descriptors. Defaults to False.

    Returns:
        bool: A flag indicating whether the merge was successful or not.
    """
    descriptors_db:Dict[str, np.ndarray] = utils.get_descriptor_database(filepath)
    flag = False
    for name_descriptor, information in descriptors.items():
        if (descriptors_db.get(name_descriptor, None) is None) or (overwrite is True):
            descriptors[name_descriptor] = information
            save_descriptor_bbdd(descriptors=descriptors, filepath=filepath)
            flag = True

    return flag

def save_descriptor_bbdd(descriptors: Dict[str, np.ndarray], filepath:str, filename:Optional[str]= None) -> None:
    """
    Save a dictionary of descriptors to the descriptor database file.

    Parameters:
        descriptors (Dict[str, np.ndarray]): A dictionary of descriptors to be saved.
        filepath (str): The directory where the database file is stored.
        filename (str, optional): The name of the database file. Defaults to None.
    """
    if filename == None:
        filename = os.path.basename(filepath)
        filepath = os.path.dirname(filepath)

    if not os.path.exists(filepath):
        create_descriptor_database(filepath=filepath, filename=filename)

    path = os.path.join(filepath, filename) if filename is not None else filepath
    print(descriptors)
    write_pickle(descriptors, path)


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

def convert2rgchromaticity(img: np.ndarray) -> np.ndarray:
    """
    Convert a linear RGB image to rg chromaticity color space.

    Parameters:
        img (np.ndarray): The input linear RGB image as a NumPy array.

    Returns:
        np.ndarray: The image in rg chromaticity color space as a NumPy array.
    """
    sum = img[:,:,0] + img[:,:,1] + img[:,:,2]
    return img / sum[:, :, None]

def normalize_min_max(x: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    r"""Normalise an image/video tensor by MinMax and re-scales the value between a range.

    The data is normalised using the following formulation:

    .. math::
        y_i = (b - a) * \frac{x_i - \text{min}(x)}{\text{max}(x) - \text{min}(x)} + a

    where :math:`a` is :math:`\text{min_val}` and :math:`b` is :math:`\text{max_val}`.

    Args:
        x: The image tensor to be normalised with shape :math:`(B, C, *)`.
        min_val: The minimum value for the new range.
        max_val: The maximum value for the new range.
        eps: Float number to avoid zero division.

    Returns:
        The normalised image tensor with same shape as input :math:`(B, C, *)`.

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if not isinstance(min_val, float):
        raise TypeError(f"'min_val' should be a float. Got: {type(min_val)}.")

    if not isinstance(max_val, float):
        raise TypeError(f"'b' should be a float. Got: {type(max_val)}.")

    if len(x.shape) < 3:
        raise ValueError(f"Input shape must be at least a 3d tensor. Got: {x.shape}.")

    x = x.unsqueeze(0) if len(x.shape) == 3 else x
    shape = x.shape
    B, C = shape[0], shape[1]

    x_min: torch.Tensor = x.reshape(B, C, -1).min(-1)[0].view(B, C, 1)
    x_max: torch.Tensor = x.reshape(B, C, -1).max(-1)[0].view(B, C, 1)

    x_out: torch.Tensor = (max_val - min_val) * (x.reshape(B, C, -1) - x_min) / (x_max - x_min + eps) + min_val
    return x_out.reshape(shape)


def read_img(img_path: Path):
    """
    Opens an image file specified by the provided file path using the PIL library.

    Args:
        img_path (Path): The path to the image file.

    Returns:
        Image.Image: A PIL Image object representing the opened image.
    """
    img = K.io.load_image(img_path, K.io.ImageLoadType.RGB32)

    return img
    #return Image.open(img_path)


def transform_tiles_colorspace(tiles: List[Image], colorspace: Callable) -> List[np.ndarray]:
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
        img = utils.normalize_min_max(img)
        img = colorspace(img)
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
    channels = [split_image_colorspace(tile) for tile in tiles]
    channel_arrays = tuple(np.array(channel) for channel in zip(*channels))
    return channel_arrays


# Example usage
# Assume 'tiles' is a list of N-dimensional color tiles
# channel_arrays = split_nd_tiles_colorspace(tiles)
# channel_arrays will contain the N arrays representing individual color channels

def split_image_colorspace(image: np.ndarray) -> Tuple[np.ndarray]:
    """
    Split an N-dimensional image into individual color channels.

    Parameters:
        image (np.ndarray): A NumPy array representing an N-dimensional color image.

    Returns:
        Tuple[np.ndarray]: A tuple containing N arrays representing individual color channels.
    """
    # Split the image into color channels
    channels = np.split(image, image.shape[-1], axis=-1)
    return tuple(channels)

# Example usage
# Assume 'image' is a 3-dimensional color image (e.g., height x width x channels)
# channel_arrays = split_image_colorspace(image)
# channel_arrays will contain the 3 arrays representing individual color channels






