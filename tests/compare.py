import numpy as np
import os
from helper import (check_close_equal, write_equal, print_close_equal) 


def compare():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"
    path_a = test_dir_path + "/a.npy"
    path_b = test_dir_path + "/b.npy"

    a = np.load(path_a)
    b = np.load(path_b)

    ratio_close, ratio_equal = check_close_equal(a, b)
    print_close_equal("Ratio", ratio_close, ratio_equal)

    path = test_dir_path + "/value.npy"
    write_equal(ratio_close, path)


if __name__ == "__main__":
    compare()

