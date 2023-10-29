from math import log10, sqrt
from utils import utils
from preprocessing import pipelines as PP
from methods import Colors_Descriptors as CD
from metrics import retrieval_distances as rd
from scipy import ndimage as ndi
import numpy as np
import kornia as K
import cv2
import matplotlib.pyplot as plt 
import pickle
from metrics import retrieval_metrics
from PIL import Image
from scipy.signal import convolve2d
from tqdm import tqdm
from skimage import measure

from pathlib import  Path

def convert2lab(img: np.ndarray):
    return (cv2.cvtColor(img, cv2.COLOR_BGR2LAB))


def get_piramidal_color_descriptor(img: np.array, steps:int=4, **kwargs):
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
            hist, _, _ = np.histogram2d(img[i:i+k_size_i, j:j+k_size_j, 1].flatten(), img[i:i+k_size_i, j:j+k_size_j, 2].flatten(), **kwargs)
            feature = np.concatenate((feature,(hist/np.sum(hist)).flatten()), axis=-1)

    return feature


def find_psnr(img, cleaned):
    """
    Finds the psnr between the original and the cleaned images
    returns:
        psnr -> float
    """
    mse = np.mean((img - cleaned) ** 2)
    
    if mse == 0:
        return 100
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))

    return psnr

def median(img):
    """
    Apply median filter
    returns:
        np.array
    """
    return cv2.medianBlur(img, 3)

def gaussian(img):
    """
    Apply gaussian filter
    returns:
        np.array
    """
    k = 3
    return cv2.GaussianBlur(img, (k, k), 0)

def bilateral(img):
    """
    Apply bilatera filtering
    returns:
        np.array
    """
    return cv2.bilateralFilter(img, 9, 75, 75)

def gabor_kernels():
    """
    Create gabor kernels
    returns:
        list with gabor kernels
    """

    kernel_list = []

    # range of orientation values 
    theta_range = np.arange(0, np.pi, np.pi / 16)  

    
    # for loop for generating different rotation 
    for theta in theta_range: 
        # kernel size 
        ksize = 35
        # sigma for Gaussian envelope 
        sigma = 1.5  
        # frequency of sinusoidal wave 
        lambd = .3
        # phase of sinusoidal wave 
        phase = 0
        gamma = .5
        kernel = cv2.getGaborKernel((ksize, ksize), 
                                    sigma, theta, 
                                    lambd, gamma, phase) 
        # kernel /= 1.0 * kernel.sum()
        kernel_list.append(kernel)
        # plt.imshow(kernel, 
        #         cmap='gray') 
        # plt.title("Kernel with orientation = " + str(theta)) 
        # plt.show()

    # theta_range = np.arange(0, np.pi, np.pi / 16)  

    # for theta in theta_range: 
    #     # kernel size 
    #     ksize = 20
    #     # sigma for Gaussian envelope 
    #     sigma = 3  
    #     # frequency of sinusoidal wave 
    #     lambd = .3
    #     # phase of sinusoidal wave 
    #     phase = 0
    #     gamma = 0
    #     kernel = cv2.getGaborKernel((ksize, ksize), 
    #                                 sigma, theta, 
    #                                 lambd, gamma, phase) 
    #     # kernel /= 1.0 * kernel.sum()
    #     kernel_list.append(kernel)
        # plt.imshow(kernel, 
        #         cmap='gray') 
        # plt.title("Kernel with orientation = " + str(theta)) 
        # plt.show()

    return kernel_list

def new_kernel():

    filters = []

    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 3 # Larger Values produce more edges
    lambd = 10
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results

    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization  
        filters.append(kern)
    
        # plt.imshow(kern, 
        #         cmap='gray') 
        # plt.title("Kernel with orientation = " + str(theta)) 
        # plt.show()

    return filters

# % Find the noise in the red.
# noiseImage = (redChannel == 0 | redChannel == 255);
# % Get rid of the noise in the red by replacing with median.
# noiseFreeRed = redChannel;
# noiseFreeRed(noiseImage) = redMF(noiseImage);

def denoise(img):

    (r, g, b) = cv2.split(img)

    # w, h, _ = img.shape
    # row, col = slice(0, w), slice(0, h)

    # if r[row, col].any(0) or r[row, col].any(255):
    #     red = median(r)
    
    # if g[row, col].any(0) or g[row, col].any(266):
    #     green = median(g)

    # if b.any(0) or b.any(266):
    #     blue = median(b)
    red = gaussian(median(r))
    green = gaussian(median(g))
    blue = gaussian(median(b))

    return cv2.merge((red, green, blue))

# def power(image, kernel):
#     # Normalize images for better comparison.
#     image = (image - image.mean()) / image.std()
#     return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
#                    ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

def estimate_noise(I):

    H, W = I.shape

    M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))

    return sigma

def CDenoissing(image):
    
    (r, g, b) = cv2.split(image)

    b_blur = cv2.medianBlur(b, 3)
    g_blur = cv2.medianBlur(g, 3)
    r_blur = cv2.medianBlur(r, 3)

    b_denoise = cv2.fastNlMeansDenoising(b_blur, 5, 9, 21)
    g_denoise = cv2.fastNlMeansDenoising(g_blur, 5, 9, 21)
    r_denoise = cv2.fastNlMeansDenoising(r_blur, 5, 9, 21)

    enhanced = cv2.merge((r_denoise, g_denoise, b_denoise))
    
    
    return enhanced

def apply_filter(img, filters):
# This general function is designed to apply filters to our image
     
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
     
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image
     
    for kern in filters:  # Loop through the kernels in our GaborFilter
        # plt.imshow(kern)
        # plt.show()
        image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
         
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage

def get_gabor_filter_descriptor(img, g_ker):
    
    min_interval = 100
    max_interval = 250
        
    fimg = apply_filter(img, g_ker)
    # cv2.imshow("a",fimg)
    # cv2.waitKey()
 
    norm_fimg = cv2.normalize(fimg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    formated_image = (norm_fimg * 255).astype("uint8")
    # cv2.imshow("gabor", fimg)
    # cv2.waitKey()
    image_edge_g = cv2.Canny(formated_image, min_interval, max_interval)

    # cv2.imshow("b",image_edge_g)
    # cv2.waitKey()

    pyramid_canny = CD.get_piramidal_histogram_descriptor(image_edge_g, steps=5)

    return pyramid_canny




def main():
    query_images = sorted(utils.read_bbdd("data/qsd2_w3"))
    bank_images = utils.read_bbdd("data/BBDD/BBDD")

    dic_combined_descriptors = {}
    bank_combined_descriptors = {}

    # kernel_lists = gabor_kernels()
    kernel_lists = new_kernel()

    # paints = utils.read_pickle(filepath="src\painting_extraction_qsd2_w3.pkl")

    paints = []
    query_folder = Path('data\qst2_w3')
    for file in query_folder.glob('masks/*.png'):

        results = []

        image = np.array(Image.open(query_folder / (Path(file).stem + '.jpg')))
        mask = np.array(Image.open(file))

        labels = measure.label(mask)

        for reg in measure.regionprops(labels):
            if reg.label == 0: 
                continue
            results.append(image[reg.bbox[0]:reg.bbox[2], reg.bbox[1]:reg.bbox[3]])
        paints.append({'result': results})
    

    for idx, dic in enumerate(paints):
        list_img = dic['result']
        for jdx, paint in enumerate(list_img):
            cleaned_median = denoise(paint)
            texture_descriptor = get_gabor_filter_descriptor(cleaned_median, kernel_lists)

            denoised = convert2lab(cleaned_median)
            color_descriptor = get_piramidal_color_descriptor(denoised)
            dic_combined_descriptors[(idx, jdx)] = np.concatenate((texture_descriptor, color_descriptor))
            
            print(idx)

    # exit()
    for j in bank_images:

        im = utils.read_img(j)

        texture_descriptor = get_gabor_filter_descriptor(im, kernel_lists)
        im = convert2lab(im)
        color_descriptor = get_piramidal_color_descriptor(denoised)
        bank_combined_descriptors[j] = np.concatenate((texture_descriptor, color_descriptor))
        print(j)

    response = PP.generate_K_response(bank_combined_descriptors, dic_combined_descriptors, sim_func=rd.histogram_intersection, k=10)
    utils.write_pickle(response, "./resultado_pickel_wk3.pkl")
    
if __name__ == "__main__":
    main()
    # gabor_kernels()
