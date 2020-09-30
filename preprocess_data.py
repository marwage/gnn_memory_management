import json
import numpy as np
import scipy.sparse as sp
import scipy.io.harwell_boeing as hb
import os

home = os.getenv("HOME")
dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr"

path = dir_path + "/adj_full.npz"
f = np.load(path)
adj = sp.csr_matrix((f["data"], f["indices"], f["indptr"]), f["shape"])
adj = adj.astype(np.float32)
path = dir_path + "/adjacency.hb"
hb.hb_write(path, adj)

path = dir_path + "/feats.npy"
features = np.load(path)
features = features.astype(np.float32)
path = dir_path + "/features.npy"
np.save(path, features)
[print(x[0:10]) for x in features[0:10]]

classes = np.zeros((features.shape[0],), dtype=np.int32)
path = dir_path + "/class_map.json"
with open(path) as f:
    class_map = json.load(f)
    for key, item in class_map.items():
        classes[int(key)] = item
path = dir_path + "/classes.npy"
np.save(path, classes)
[print(x) for x in classes[0:10]]

path = dir_path + "/role.json"
with open(path) as f:
    role = json.load(f)
train_mask = np.zeros((features.shape[0],), dtype=bool)
train_mask[np.array(role["tr"])] = True
path = dir_path + "/train_mask.npy"
np.save(path, train_mask)
[print(x) for x in train_mask[0:10]]
val_mask = np.zeros((features.shape[0],), dtype=bool)
val_mask[np.array(role["va"])] = True
path = dir_path + "/val_mask.npy"
np.save(path, val_mask)
[print(x) for x in val_mask[0:10]]
test_mask = np.zeros((features.shape[0],), dtype=bool)
test_mask[np.array(role["te"])] = True
path = dir_path + "/test_mask.npy"
np.save(path, test_mask)
[print(x) for x in test_mask[0:10]]

