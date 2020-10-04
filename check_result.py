import json
import numpy as np
import scipy.sparse as sp
import scipy.io
import os

home = os.getenv("HOME")
dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr"

path = dir_path + "/adjacency.mtx"
adj = scipy.io.mmread(path)

path = dir_path + "/features.npy"
features = np.load(path)

path = dir_path + "/result.npy"
result = np.load(path)

true_result = adj.dot(features)

print("Number of non zero elements: {}".format(np.count_nonzero(result)))
assert(result.shape == true_result.shape)
percentage_equal = (result.shape[0] * result.shape[1]) / ((result == true_result).sum())
print("Percentage of equal elements: {}".format(percentage_equal))

