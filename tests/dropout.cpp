// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>


const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";


int test_dropout(int chunk_size) {
    // read features
    std::string path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    CudaHelper cuda_helper;
    DropoutParent *dropout_layer;
    if (chunk_size == 0) {// no chunking
        dropout_layer = new Dropout(&cuda_helper, features.num_rows_, features.num_columns_);
    } else {
        dropout_layer = new DropoutChunked(&cuda_helper, chunk_size, features.num_rows_, features.num_columns_);
    }

    Matrix<float> *dropout_result = dropout_layer->forward(&features);
    path = test_dir_path + "/dropout_result.npy";
    save_npy_matrix(dropout_result, path);

    Matrix<float> in_gradients;
    in_gradients.num_rows_ = features.num_rows_;
    in_gradients.num_columns_ = features.num_columns_;
    in_gradients.values_ = reinterpret_cast<float *>(
            malloc(in_gradients.num_rows_ * in_gradients.num_columns_ * sizeof(float)));
    for (int i = 0; i < in_gradients.num_rows_ * in_gradients.num_columns_; ++i) {
        in_gradients.values_[i] = rand();
    }
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&in_gradients, path);

    Matrix<float> *gradients = dropout_layer->backward(&in_gradients);
    path = test_dir_path + "/dropout_gradients.npy";
    save_npy_matrix(gradients, path);

    // compare with Pytorch, Numpy, SciPy
    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/dropout.py";
    system(command);

    // TODO Due to randomness never 1
    path = test_dir_path + "/value.npy";
    int return_value = read_return_value(path);
    return 1;
}


TEST_CASE("Dropout", "[dropout]") {
    CHECK(test_dropout(0));
}

TEST_CASE("Dropout chunked", "[dropout][chunked]") {
    CHECK(test_dropout(1 << 15));
    CHECK(test_dropout(1 << 12));
    CHECK(test_dropout(1 << 8));
}
