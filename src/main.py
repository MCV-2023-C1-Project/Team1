# noinspection PyInterpreter
from utils import utils
from methods import Colors_Descriptors as CD
from preprocessing import pipelines as pipe

from math import ceil
from image_slicer import slice
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

import matplotlib.pyplot as plt


import kornia as K
import kornia.enhance as ke
import numpy as np
import cv2

import pickle as pkl




import torch
import time


with open("data/qsd1_w1/gt_corresps.pkl", "rb") as f:
    file = pkl.load(f)
    print(file)


DESCRIPTORS_PATH = "data/BBDD/color/descriptors.pkl"


img_list = sorted(utils.read_bbdd("data/BBDD"))

img_test = str(img_list[277])
descriptors_dict = {}
dic = utils.get_descriptor_database(DESCRIPTORS_PATH)

query_tiles = utils.get_slices_from_image(img_test, 16)
dif_colors = pipe.transform_tiles_colorspace(query_tiles, colorspace=utils.convert2lab)
_, U_query, V_Query = pipe.split_nd_tiles_colorspace(dif_colors)

histogram_U, pdf_U_query = CD.get_multi_tile_histogram_descriptor(imgs=U_query, n_bins = 32, return_pdf = True)
histogram_V, pdf_V_query = CD.get_multi_tile_histogram_descriptor(imgs=V_Query, n_bins = 32, return_pdf = True)

U_descriptor_query = np.array([])
V_descriptor_query = np.array([])

for i,j in zip(pdf_U_query, pdf_V_query):
    U_descriptor_query = np.concatenate((U_descriptor_query, i), axis=-1)
    V_descriptor_query = np.concatenate((V_descriptor_query, j), axis=-1)

query_descriptor = U_descriptor_query + V_descriptor_query

results_list = []
descriptors_dict = {}
## Comparing with all the images
for idx, img_train in tqdm(enumerate(img_list)):

    img_tiles = utils.get_slices_from_image(img_train, 16)
    dif_colors = pipe.transform_tiles_colorspace(img_tiles, colorspce=utils.convert2lab)
    _, U, V = pipe.split_nd_tiles_colorspace(dif_colors)

    histogram_U, pdf_U = CD.get_multi_tile_histogram_descriptor(tiles=U, n_bins=32, return_pdf=True)
    histogram_V, pdf_V = CD.get_multi_tile_histogram_descriptor(tiles=V, n_bins=32, return_pdf=True)

    U_descriptor = np.array([])
    V_descriptor = np.array([])

    for i, j in zip(pdf_U, pdf_V):
        U_descriptor = np.concatenate((U_descriptor, i), axis=-1)
        V_descriptor = np.concatenate((V_descriptor, j), axis=-1)

    img_descriptor = U_descriptor + V_descriptor

    descriptors_dict[img_train.name] = img_descriptor


    distance = 1-jensenshannon(img_descriptor, query_descriptor)

    results_list.append((distance, img_train.name))

exit()
print(results_list)


results_sorted = sorted(results_list, key=lambda x: x[0], reverse=True)
print(results_sorted)

with open("results.txt", "a") as res:
    print("Writing results")
    res.write("Testing on img 170:\n")
    res.write("The Top 10 imatges are:\n")
    for resul in results_sorted[:10]:
        res.write(f"{resul}\n")














