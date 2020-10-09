// Copyright 2020 Marcel Wagenländer

#include "cuda_helper.hpp"
#include "graph_convolution.hpp"
#include "tensors.hpp"
#include "dropout.hpp"
#include "sage_linear.hpp"
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
    to_column_major<float>(&features);

    // read classes
    path = dir_path + "/classes.npy";
    matrix<int> classes = load_npy_matrix<int>(path);

    // read train_mask
    path = dir_path + "/train_mask.npy";
    matrix<bool> train_mask = load_npy_matrix<bool>(path);

    // read val_mask
    path = dir_path + "/val_mask.npy";
    matrix<bool> val_mask = load_npy_matrix<bool>(path);

    // read test_mask
    path = dir_path + "/test_mask.npy";
    matrix<bool> test_mask = load_npy_matrix<bool>(path);

    // read adjacency
    path = dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    // FORWARD PASS
    CudaHelper cuda_helper;

    // dropout 0
    Dropout dropout(&cuda_helper);
    matrix<float> dropout_result_0 = dropout.forward(features);

    // graph convolution 0
    GraphConvolution graph_convolution(&cuda_helper);
    matrix<float> graph_conv_result_0 = graph_convolution.forward(adjacency,
                                                                  dropout_result_0, "mean");

    // DEBUG
    matrix<float> graph_conv_result = graph_convolution.forward(adjacency, features, "sum");
    path = dir_path + "/graph_conv_result.npy";
    save_npy_matrix(graph_conv_result, path);
    matrix<float> graph_conv_mean_result = graph_convolution.forward(adjacency, features, "mean");
    path = dir_path + "/graph_conv_mean_result.npy";
    save_npy_matrix(graph_conv_mean_result, path);
    // END DEBUG

    // linear layer 0
    int num_hidden_channels = 128;
    SageLinear linear_0(features.columns, num_hidden_channels, &cuda_helper);
    matrix<float> linear_result_0 = linear_0.forward(dropout_result_0,
                                                     graph_conv_result_0);

    // ReLU 0
    Relu relu(&cuda_helper);
    matrix<float> relu_result_0 = relu.forward(linear_result_0);

    // dropout 1
    matrix<float> dropout_result_1 = dropout.forward(relu_result_0);

    // graph convolution 1
    matrix<float> graph_conv_result_1 = graph_convolution.forward(adjacency,
                                                                  dropout_result_1, "mean");

    // linear layer 1
    SageLinear linear_1(num_hidden_channels, num_hidden_channels, &cuda_helper);
    matrix<float> linear_result_1 = linear_1.forward(dropout_result_1,
                                                     graph_conv_result_1);

    // ReLU 0
    matrix<float> relu_result_1 = relu.forward(linear_result_1);

    // dropout 2
    matrix<float> dropout_result_2 = dropout.forward(relu_result_1);

    // graph convolution 2
    matrix<float> graph_conv_result_2 = graph_convolution.forward(adjacency,
                                                                  dropout_result_2, "mean");

    // linear layer 2
    int num_classes = 7;
    SageLinear linear_2(num_hidden_channels, num_classes, &cuda_helper);
    matrix<float> linear_result_2 = linear_2.forward(dropout_result_2,
                                                     graph_conv_result_2);

    // softmax
    LogSoftmax log_softmax(&cuda_helper);
    matrix<float> softmax_result = log_softmax.forward(linear_result_2);

    // loss
    float loss = negative_log_likelihood_loss(softmax_result, classes);
    std::cout << "loss " << loss << std::endl;

    // destroy cuda handles
    cuda_helper.destroy_handles();

    // free memory
    free(features.values);
    free(classes.values);
    free(train_mask.values);
    free(val_mask.values);
    free(test_mask.values);
}

