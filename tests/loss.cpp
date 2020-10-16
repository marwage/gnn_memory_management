// Copyright 2020 Marcel Wagenländer

#include <iostream>

#include "tensors.hpp"
#include "loss.hpp"


void check_loss() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr";

    std::string path = dir_path + "/log_softmax_out.npy";
    matrix<float> log_softmax_out = load_npy_matrix<float>(path);
    path = dir_path + "/classes.npy";
    matrix<int> classes = load_npy_matrix<int>(path);

    NLLLoss loss_layer;
    float loss = loss_layer.forward(log_softmax_out, classes);
    std::cout << "loss " << loss << std::endl;

    matrix<float> grads = loss_layer.backward();

    path = dir_path + "/loss_grads.npy";
    save_npy_matrix(grads, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/loss.py";
    system(command);
}

int main() {
    check_loss();
}
