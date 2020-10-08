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

    // FORWARD PASS
    // dropout 0
    matrix<float> dropout_result_0 = dropout(features);

    // graph convolution 0
    matrix<float> graph_conv_result_0 = graph_convolution(adjacency, dropout_result_0, "mean");

    // linear layer 0
    int num_hidden_channels = 128;
    Linear linear_0(features.columns, num_hidden_channels);
    matrix<float> linear_result_0 = linear_0.forward(graph_conv_result_0);

    // ReLU 0
    matrix<float> relu_result_0 = relu(linear_result_0);

    // dropout 1
    matrix<float> dropout_result_1 = dropout(relu_result_0);

    // graph convolution 1
    matrix<float> graph_conv_result_1 = graph_convolution(adjacency, dropout_result_1, "mean");

    // linear layer 1
    Linear linear_1(num_hidden_channels, num_hidden_channels);
    matrix<float> linear_result_1 = linear_1.forward(graph_conv_result_1);

    // dropout 2
    matrix<float> dropout_result_2 = dropout(linear_result_1);

    // graph convolution 2
    matrix<float> graph_conv_result_2 = graph_convolution(adjacency, dropout_result_2, "mean");

    // linear layer 2
    int num_classes = 7;
    Linear linear_2(num_hidden_channels, num_classes);
    matrix<float> linear_result_2 = linear_2.forward(graph_conv_result_2);

    // softmax
    matrix<float> softmax_result = softmax(linear_result_2);

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

