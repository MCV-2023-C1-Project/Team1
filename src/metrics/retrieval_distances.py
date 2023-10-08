import numpy as np
from scipy.spatial.distance import jensenshannon, cosine

def euclDis(u:np.array, v:np.array):

    return np.linalg.norm(u, v)

def l1norm(u:np.array, v:np.array):
    return np.linalg.norm(u,v, ord=1)

def chi2_distance(u:np.array, v:np.array):
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                        for (a, b) in zip(u, v)])
    return chi
def hellkdis(u:np.array, v:np.array):
    distance = np.sum([np.sqrt(a * b)
                    for (a, b) in zip(u, v)])
    return distance

def histogram_intersection(u:np.array, v:np.array):
    return np.sum(np.minimum(u, v))

def jensensim(v:np.array, u:np.array):
    return 1-jensenshannon(v, u)

def cos_sim(v:np.array, u:np.array):
    return 1-abs(cosine(v,u))
