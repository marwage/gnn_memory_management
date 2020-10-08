// Copyright 2020 Marcel Wagenländer

#include "cuda_helper.hpp"
#include "graph_convolution.hpp"
#include "tensors.hpp"
#include "dropout.hpp"
#include "linear.hpp"
#include "activation.hpp"
#include "loss.hpp"

#include "cnpy.h"


int main() {
    // read tensors
    // set path to directory
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr";

    // read features
    std::string path = dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);

    // read classes
    path = dir_path + "/classes.npy";
    vector<int> classes = load_npy_vector<int>(path);

    // read train_mask
    path = dir_path + "/train_mask.npy";
    vector<bool> train_mask = load_npy_vector<bool>(path);

    // read val_mask
    path = dir_path + "/val_mask.npy";
    vector<bool> val_mask = load_npy_vector<bool>(path);

    // read test_mask
    path = dir_path + "/test_mask.npy";
    vector<bool> test_mask = load_npy_vector<bool>(path);

    // read adjacency
    path = dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    // graph convolution
    matrix<float> result = graph_convolution(adjacency, features, "sum");
    
    matrix<float> graph_conv_result_mean = graph_convolution(adjacency, features, "mean");

    // write result to npy file
    path = dir_path + "/graph_convolution_result.npy";
    std::vector<size_t> shape = {(size_t) result.rows, (size_t) result.columns};
    cnpy::npy_save<float>(path, result.values, shape);

    // dropout
    matrix<float> dropout_result = dropout(features);

    // write dropout result to npy file
    path = dir_path + "/dropout_result.npy";
    shape = {(size_t) dropout_result.rows, (size_t) dropout_result.columns};
    cnpy::npy_save<float>(path, dropout_result.values, shape);

    // linear layer
    matrix<float> linear_result = linear(features);

    // write linear layer result to npy file
    path = dir_path + "/linear_result.npy";
    shape = {(size_t) linear_result.rows, (size_t) linear_result.columns};
    cnpy::npy_save<float>(path, linear_result.values, shape);

    // ReLU
    matrix<float> relu_result = relu(features);

    // softmax
    matrix<float> softmax_result = softmax(linear_result);

    // loss
    float loss = negative_log_likelihood_loss(softmax_result, classes);

    std::cout << "loss " << loss << std::endl;

    // free memory
    free(features.values);
    free(classes.values);
    free(train_mask.values);
    free(val_mask.values);
    free(test_mask.values);
}

