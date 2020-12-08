// Copyright 2020 Marcel Wagenländer

#include "loss.hpp"
#include "helper.hpp"
#include "tensors.hpp"
#include "chunking.hpp"

#include "catch2/catch.hpp"
#include <string>
#include <iostream>

const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";


int test_loss() {
    int num_classes = 7;

    std::string path = flickr_dir_path + "/classes.npy";
    Matrix<int> classes = load_npy_matrix<int>(path);
    Matrix<float> input(classes.num_rows_, num_classes, true);
    input.set_random_values();
    path = test_dir_path + "/input.npy";
    save_npy_matrix(&input, path);

    NLLLoss loss_layer(input.num_rows_, input.num_columns_);

    float loss = loss_layer.forward(&input, &classes);
    Matrix<float> loss_mat;
    loss_mat.num_rows_ = 1;
    loss_mat.num_columns_ = 1;
    loss_mat.values_ = new float[1];
    loss_mat.values_[0] = loss;
    path = test_dir_path + "/loss.npy";
    save_npy_matrix(&loss_mat, path);

    Matrix<float> *gradients = loss_layer.backward();
    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(gradients, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/loss.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_loss_chunked(long chunk_size) {
    int num_classes = 7;

    std::string path = flickr_dir_path + "/classes.npy";
    Matrix<int> classes = load_npy_matrix<int>(path);
    Matrix<float> input(classes.num_rows_, num_classes, true);
    input.set_random_values();
    path = test_dir_path + "/input.npy";
    save_npy_matrix(&input, path);

    long num_chunks = ceil((double) input.num_rows_ / (double) chunk_size);
    std::vector<Matrix<float>> input_chunked(num_chunks);
    chunk_up(&input, &input_chunked, chunk_size);

    NLLLoss loss_layer(input.num_rows_, input.num_columns_);

    float loss = loss_layer.forward(&input_chunked, &classes);
    std::cout << "Loss: " << loss << std::endl;
    Matrix<float> loss_mat;
    loss_mat.num_rows_ = 1;
    loss_mat.num_columns_ = 1;
    loss_mat.values_ = new float[1];
    loss_mat.values_[0] = loss;
    path = test_dir_path + "/loss.npy";
    save_npy_matrix(&loss_mat, path);

    Matrix<float> *gradients = loss_layer.backward();
    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(gradients, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/loss.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}


TEST_CASE("Loss", "[loss]") {
    CHECK(test_loss());
}

TEST_CASE("Loss, chunked", "[loss][chunked]") {
    CHECK(test_loss_chunked(1 << 15));
    CHECK(test_loss_chunked(1 << 12));
    CHECK(test_loss_chunked(1 << 8));
}
