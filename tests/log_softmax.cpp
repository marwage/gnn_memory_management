// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <stdlib.h>
#include "catch2/catch.hpp"


int test_log_softmax_forward() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr";

    // read features
    std::string path = dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    to_column_major<float>(&features);

    CudaHelper cuda_helper;
    LogSoftmax log_softmax(&cuda_helper);
    matrix<float> signals = log_softmax.forward(features);

    path = dir_path + "/log_softmax_out.npy";
    save_npy_matrix(signals, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/log_softmax.py";
    system(command);

    return 1; // TODO
}


TEST_CASE("Log-softmax forward","[logsoftmax][forward]") {
    CHECK(test_log_softmax_forward());
}
