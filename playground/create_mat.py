import numpy as np
import os


def mat():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    test_dir_path = dir_path + "/tests"

    a = np.random.rand(99999, 88888)

    path = test_dir_path + "/matrixxx.npy"
    np.save(path, a)

    b = np.load(path)


if __name__ == "__main__":
    mat()

