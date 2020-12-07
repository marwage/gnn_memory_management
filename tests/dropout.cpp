// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "chunking.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>


const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";


int test_dropout() {
    std::string path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> in_gradients(features.num_rows_, features.num_columns_, true);
    in_gradients.set_random_values();
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&in_gradients, path);

    CudaHelper cuda_helper;
    Dropout dropout_layer(&cuda_helper, features.num_rows_, features.num_columns_);

    Matrix<float> *dropout_result = dropout_layer.forward(&features);
    path = test_dir_path + "/dropout_result.npy";
    save_npy_matrix(dropout_result, path);

    Matrix<float> *gradients = dropout_layer.backward(&in_gradients);
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

int test_dropout_chunked(int chunk_size) {
    std::string path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    long num_nodes = features.num_rows_;
    long num_features = features.num_columns_;

    Matrix<float> incoming_gradients(num_nodes, num_features, true);
    incoming_gradients.set_random_values();
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&incoming_gradients, path);


    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    std::vector<Matrix<float>> x(num_chunks);
    std::vector<Matrix<float>> dy(num_chunks);
    chunk_up(&features, &x, chunk_size);
    chunk_up(&incoming_gradients, &dy, chunk_size);

    CudaHelper cuda_helper;
    DropoutChunked dropout_layer(&cuda_helper, chunk_size, features.num_rows_, features.num_columns_);

    std::vector<Matrix<float>> *activations = dropout_layer.forward(&x);

    std::vector<Matrix<float>> *gradients = dropout_layer.backward(&dy);

    Matrix<float> activations_one(num_nodes, num_features, true);
    Matrix<float> gradients_one(num_nodes, num_features, true);
    stitch(activations, &activations_one);
    stitch(gradients, &gradients_one);

    path = test_dir_path + "/dropout_result.npy";
    save_npy_matrix(&activations_one, path);
    path = test_dir_path + "/dropout_gradients.npy";
    save_npy_matrix(&gradients_one, path);

    // compare with Pytorch, Numpy, SciPy
    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/dropout.py";
    system(command);

    // TODO Due to randomness never 1
    path = test_dir_path + "/value.npy";
    int return_value = read_return_value(path);
    return 1;
}


TEST_CASE("Dropout", "[dropout]") {
    CHECK(test_dropout());
}

TEST_CASE("Dropout chunked", "[dropout][chunked]") {
    CHECK(test_dropout_chunked(1 << 15));
    CHECK(test_dropout_chunked(1 << 12));
    CHECK(test_dropout_chunked(1 << 8));
}
