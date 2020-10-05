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

print("features")
for i in range(10):
    print(features[i, 0:10])

path = dir_path + "/result.npy"
result = np.load(path)

true_result = adj.dot(features)

print("true_result")
for i in range(10):
    print(true_result[i, 0:10])

print("Result number of elements: {}".format(result.size))
print("Result number of non zero elements: {}".format(np.count_nonzero(result)))
assert(result.shape == true_result.shape)
percentage_equal = result.size / ((result == true_result).sum())
print("Percentage of equal elements: {}".format(percentage_equal))

