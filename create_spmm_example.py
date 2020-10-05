import numpy as np
import scipy.sparse as sp
import scipy.io
import os


A = np.array([[1, 0, 2,3],
    [0, 4, 0, 0],
    [5, 0, 6, 7],
    [0, 8, 0, 9]])
A = A.astype(np.float32)

B = np.array([[1, 5, 9],
    [2, 6, 10],
    [3, 7, 11],
    [4, 8, 12]])
B = B.astype(np.float32)

print("A")
print(A)
print("B")
print(B)

A_sp = sp.csr_matrix(A)

home = os.getenv("HOME")
dir_path = home + "/gpu_memory_reduction/alzheimer/data/example"

path = dir_path + "/A.mtx"
scipy.io.mmwrite(path, A_sp)

path = dir_path + "/B.npy"
np.save(path, B)

C = A.dot(B)

print("A dot B")
print(C)

