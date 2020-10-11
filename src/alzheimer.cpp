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
    Dropout dropout_0(&cuda_helper);
    matrix<float> dropout_result_0 = dropout_0.forward(features);

    // graph convolution 0
    GraphConvolution graph_convolution_0(&cuda_helper, &adjacency, "mean");
    matrix<float> graph_conv_result_0 = graph_convolution_0.forward(dropout_result_0);

    // linear layer 0
    int num_hidden_channels = 128;
    SageLinear linear_0(features.columns, num_hidden_channels, &cuda_helper);
    matrix<float> linear_result_0 = linear_0.forward(dropout_result_0,
                                                     graph_conv_result_0);

    // ReLU 0
    Relu relu_0(&cuda_helper);
    matrix<float> relu_result_0 = relu_0.forward(linear_result_0);

    // dropout 1
    Dropout dropout_1(&cuda_helper);
    matrix<float> dropout_result_1 = dropout_1.forward(relu_result_0);

    // graph convolution 1
    GraphConvolution graph_convolution_1(&cuda_helper, &adjacency, "mean");
    matrix<float> graph_conv_result_1 = graph_convolution_1.forward(dropout_result_1);

    // linear layer 1
    SageLinear linear_1(num_hidden_channels, num_hidden_channels, &cuda_helper);
    matrix<float> linear_result_1 = linear_1.forward(dropout_result_1,
                                                     graph_conv_result_1);

    // ReLU 1
    Relu relu_1(&cuda_helper);
    matrix<float> relu_result_1 = relu_1.forward(linear_result_1);

    // dropout 2
    Dropout dropout_2(&cuda_helper);
    matrix<float> dropout_result_2 = dropout_2.forward(relu_result_1);

    // graph convolution 2
    GraphConvolution graph_convolution_2(&cuda_helper, &adjacency, "mean");
    matrix<float> graph_conv_result_2 = graph_convolution_2.forward(dropout_result_2);

    // linear layer 2
    int num_classes = 7;
    SageLinear linear_2(num_hidden_channels, num_classes, &cuda_helper);
    matrix<float> linear_result_2 = linear_2.forward(dropout_result_2,
                                                     graph_conv_result_2);

    // log-softmax
    LogSoftmax log_softmax(&cuda_helper);
    matrix<float> softmax_result = log_softmax.forward(linear_result_2);

    // loss
    NLLLoss loss_layer;
    float loss = loss_layer.forward(softmax_result, classes);
    std::cout << "loss " << loss << std::endl;

    // BACKPROPAGATION
    //loss
    matrix<float> gradients = loss_layer.backward();

    // log-softmax
    gradients = log_softmax.backward(gradients);

    // linear layer 2
    gradients = linear_2.backward(gradients);

    // graph convolution 2
    gradients = graph_convolution_2.backward(gradients);

    // dropout 2
    gradients = dropout_2.backward(gradients);

    // CLEAN-UP
    // destroy cuda handles
    cuda_helper.destroy_handles();

    // free memory
    free(features.values);
    free(classes.values);
    free(train_mask.values);
    free(val_mask.values);
    free(test_mask.values);
}

