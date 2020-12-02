// Copyright 2020 Marcel Wagenländer

#include "loss.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>


int test_loss() {
    std::string home = std::getenv("HOME");
    int num_classes = 7;

    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";

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


TEST_CASE("Loss", "[loss]") {
    CHECK(test_loss());
}
