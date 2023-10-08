
from utils import utils
from methods import Colors_Descriptors as CD
from preprocessing import pipelines as pipe

from math import ceil
from image_slicer import slice
from scipy.spatial.distance import jensenshannon, cosine
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



img_list = sorted(utils.read_bbdd("data/BBDD"))
img_test_list = sorted(utils.read_bbdd("data/qsd1_w1"))

img_test = str(img_test_list[0])
img_test_2 = str(img_list[120])


query_tiles = utils.get_slices_from_image(img_test, 16)
dif_colors = utils.transform_tiles_colorspace(query_tiles, colorspace=utils.convert2lab)
_, U_query, V_Query = utils.split_nd_tiles_colorspace(dif_colors)

#U_query = U_query.squeeze()
#V_Query = V_Query.squeeze()

histogram_U, pdf_U_query = CD.get_multi_tile_histogram_descriptor(tiles=U_query, n_bins = 64, return_pdf = True)
histogram_V, pdf_V_query = CD.get_multi_tile_histogram_descriptor(tiles=V_Query, n_bins = 64, return_pdf = True)

U_descriptor_query = np.zeros(64)
V_descriptor_query = np.zeros(64)

for i,j in zip(pdf_U_query, pdf_V_query):
    U_descriptor_query += i
    V_descriptor_query += j

query_descriptor = U_descriptor_query + V_descriptor_query
print(query_descriptor)


## image test 2
query_tiles = utils.get_slices_from_image(img_test_2, 16)
dif_colors = utils.transform_tiles_colorspace(query_tiles, colorspace=utils.convert2lab)
_, U_query, V_Query = utils.split_nd_tiles_colorspace(dif_colors)

#U_query = U_query.squeeze()
#V_Query = V_Query.squeeze()

histogram_U, pdf_U_query = CD.get_multi_tile_histogram_descriptor(tiles=U_query, n_bins = 64, return_pdf = True)
histogram_V, pdf_V_query = CD.get_multi_tile_histogram_descriptor(tiles=V_Query, n_bins = 64, return_pdf = True)

U_descriptor_query = np.zeros(64)
V_descriptor_query = np.zeros(64)

for i,j in zip(pdf_U_query, pdf_V_query):
    U_descriptor_query += i
    V_descriptor_query += j

query_descriptor_2 = U_descriptor_query + V_descriptor_query
print(query_descriptor_2)

print(1-jensenshannon(query_descriptor_2, query_descriptor))
print(1-cosine(query_descriptor_2, query_descriptor))


results_list = []

## Comparing with all the images
for idx, img_train in tqdm(enumerate(img_list)):

    img_tiles = utils.get_slices_from_image(img_train, 16)
    dif_colors = utils.transform_tiles_colorspace(img_tiles, colorspace=utils.convert2lab)
    _, U, V = utils.split_nd_tiles_colorspace(dif_colors)

    histogram_U, pdf_U = CD.get_multi_tile_histogram_descriptor(tiles=U, n_bins=64, return_pdf=True)
    histogram_V, pdf_V = CD.get_multi_tile_histogram_descriptor(tiles=V, n_bins=64, return_pdf=True)

    U_descriptor = np.zeros(64)
    V_descriptor = np.zeros(64)

    for i, j in zip(pdf_U, pdf_V):
        U_descriptor += i
        V_descriptor += j

    img_descriptor = U_descriptor + V_descriptor

    distance = 1-jensenshannon(img_descriptor, query_descriptor)

    results_list.append((distance, img_train.name))

print(results_list)

results_sorted = sorted(results_list, key=lambda x: x[0], reverse=True)
print(results_sorted)

with open("results.txt", "a") as res:
    print("Writing results")
    res.write("Testing on img 277:\n")
    res.write("The Top 10 imatges are:\n")
    for resul in results_sorted[:10]:
        res.write(f"{resul}\n")