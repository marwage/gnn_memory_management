// Copyright 2020 Marcel Wagenländer

#include "loss.hpp"
#include "chunking.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>

const std::string home = std::getenv("HOME");
const std::string flickr_dir_path = "/mnt/data/flickr";
const std::string test_dir_path = home + "/gpu_memory_reduction/alzheimer/data/tests";

int test_loss() {
    int num_classes = 7;

    std::string path = flickr_dir_path + "/classes.npy";
    Matrix<int> classes = load_npy_matrix<int>(path);
    Matrix<float> input(classes.num_rows_, num_classes, true);
    input.set_random_values();
    path = test_dir_path + "/input.npy";
    save_npy_matrix(&input, path);

    // DEBUGGING
    to_row_major_inplace(&input);

    NLLLoss loss_layer(input.num_rows_, input.num_columns_);

    float loss = loss_layer.forward(&input, &classes);
    Matrix<float> loss_mat(1, 1, true);
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

    NLLLossChunking loss_layer(input.num_rows_, input.num_columns_, chunk_size);

    float loss = loss_layer.forward(&input_chunked, &classes);
    Matrix<float> loss_mat(1, 1, true);
    loss_mat.values_[0] = loss;
    path = test_dir_path + "/loss.npy";
    save_npy_matrix(&loss_mat, path);

    std::vector<Matrix<float>> *gradients = loss_layer.backward();
    Matrix<float> gradients_one(input.num_rows_, input.num_columns_, gradients->at(0).is_row_major_);
    stitch(gradients, &gradients_one);
    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(&gradients_one, path);

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
