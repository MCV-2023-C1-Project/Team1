## import de paquetes locales (por ejemplo import methods.methods  as m )


## from ....
from typing import *

from pathlib import  Path
from PIL import Image

import os


import numpy as np





def read_bbdd(path: Type[str], x: List[int, ...]) -> List[Path, ...]:
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


def read_img(img_path: Path):
    """
    Opens an image file specified by the provided file path using the PIL library.

    Args:
        img_path (Path): The path to the image file.

    Returns:
        Image.Image: A PIL Image object representing the opened image.
    """

    return Image.open(path=img_path)





