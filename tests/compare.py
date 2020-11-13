import numpy as np
import os
from helper import check_isclose, print_small, check_equal


def compare():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"
    path_a = test_dir_path + "/a.npy"
    path_b = test_dir_path + "/b.npy"

    a = np.load(path_a)
    b = np.load(path_b)

    ratio_close = check_isclose(a, b)
    ratio_equal = check_equal(a, b)
    print("Ratio: Close: {}, Equal: {}".format(ratio_close, ratio_equal))


if __name__ == "__main__":
    compare()

