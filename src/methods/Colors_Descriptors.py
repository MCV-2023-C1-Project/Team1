from  utils import utils

from pathlib import Path
from PIL.Image import Image
from typing import *

import kornia as k
import kornia.enhance as ke
import numpy as np

def get_multi_tile_histogram_descriptor(imgs: List[np.array], **kwargs):
    """
    Compute histogram and probability density function (PDF) descriptors for multiple images.

    Parameters:
        imgs (List[np.array]): A list of NumPy arrays representing images.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Tuple[List[np.array], List[np.array]]: A tuple containing histogram and PDF descriptors.
            The histogram descriptor array and PDF descriptor array for the input images.
    """
    histogram_descriptor_array = []
    pdf_descriptor_array = []
    for img in imgs:
        im = utils.image2tensor(img)
        histogram, pdf = ke.image_histogram2d(im, **kwargs)
        histogram_descriptor_array += list(histogram.numpy())
        pdf_descriptor_array += list(pdf.numpy())

    return histogram_descriptor_array, pdf_descriptor_array


