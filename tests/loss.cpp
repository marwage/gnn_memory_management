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
    matrix<int> classes = load_npy_matrix<int>(path);
    matrix<float> input = gen_rand_matrix(classes.rows, num_classes);
    path = test_dir_path + "/input.npy";
    save_npy_matrix(input, path);

    NLLLoss loss_layer;

    float loss = loss_layer.forward(input, classes);
    matrix<float> loss_mat;
    loss_mat.rows = 1;
    loss_mat.columns = 1;
    loss_mat.values = new float[1];
    loss_mat.values[0] = loss;
    path = test_dir_path + "/loss.npy";
    save_npy_matrix(loss_mat, path);

    matrix<float> gradients = loss_layer.backward();
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
