// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "cuda_helper.hpp"
#include "dropout.hpp"
#include "graph_convolution.hpp"
#include "loss.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"


int integration_test_subset() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";

    // read features
    std::string path = flickr_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    to_column_major<float>(&features);

    // read classes
    path = flickr_dir_path + "/classes.npy";
    matrix<int> classes = load_npy_matrix<int>(path);

    // read adjacency
    path = flickr_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    // FORWARD PASS
    CudaHelper cuda_helper;
    float learning_rate = 0.003;
    int num_hidden_channels = 128;
    int num_classes = 7;

    // layers
    Dropout dropout_layer(&cuda_helper);
    GraphConvolution graph_convolution_layer(&cuda_helper, &adjacency, "mean");
    SageLinear linear_layer(features.columns, num_hidden_channels, &cuda_helper);
    Relu relu_layer(&cuda_helper);
    LogSoftmax log_softmax_layer(&cuda_helper);
    NLLLoss loss_layer;

    // generate random inputs
    matrix<float> A;
    A.rows = features.rows;
    A.columns = features.columns;
    A.values = reinterpret_cast<float *>(malloc(A.rows * A.columns * sizeof(float)));
    for (int i = 0; i < A.rows * A.columns; ++i) {
        A.values[i] = rand();
    }
    matrix<float> B;
    B.rows = features.rows;
    B.columns = features.columns;
    B.values = reinterpret_cast<float *>(malloc(B.rows * B.columns * sizeof(float)));
    for (int i = 0; i < B.rows * B.columns; ++i) {
        B.values[i] = rand();
    }
    path = test_dir_path + "/A.npy";
    save_npy_matrix(A, path);
    path = test_dir_path + "/B.npy";
    save_npy_matrix(B, path);


    // linear layer
    matrix<float> linear_result = linear_layer.forward(A, B);
    path = test_dir_path + "/linear_result.npy";
    save_npy_matrix(linear_result, path);
    matrix<float> *parameters = linear_layer.get_parameters();
    path = test_dir_path + "/self_weight.npy";
    save_npy_matrix(parameters[0], path);
    path = test_dir_path + "/self_bias.npy";
    save_npy_matrix(parameters[1], path);
    path = test_dir_path + "/neigh_weight.npy";
    save_npy_matrix(parameters[2], path);
    path = test_dir_path + "/neigh_bias.npy";
    save_npy_matrix(parameters[3], path);

    // log-softmax
    matrix<float> log_softmax_result = log_softmax_layer.forward(linear_result);
    path = test_dir_path + "/log_softmax_result.npy";
    save_npy_matrix(log_softmax_result, path);

    // loss
    float loss = loss_layer.forward(log_softmax_result, classes);
    matrix<float> loss_mat;
    loss_mat.rows = 1;
    loss_mat.columns = 1;
    loss_mat.values = &loss;
    path = test_dir_path + "/loss_result.npy";
    save_npy_matrix(loss_mat, path);

    // BACKPROPAGATION
    //loss
    matrix<float> loss_grads = loss_layer.backward();
    path = test_dir_path + "/loss_grads.npy";
    save_npy_matrix(loss_grads, path);

    // log-softmax
    matrix<float> log_softmax_grads = log_softmax_layer.backward(loss_grads);
    path = test_dir_path + "/log_softmax_grads.npy";
    save_npy_matrix(log_softmax_grads, path);

    // linear layer
    SageLinear::SageLinearGradients linear_grads = linear_layer.backward(log_softmax_grads);
    path = test_dir_path + "/self_grads.npy";
    save_npy_matrix(linear_grads.self_grads, path);
    path = test_dir_path + "/neigh_grads.npy";
    save_npy_matrix(linear_grads.neigh_grads, path);
    matrix<float> *gradients = linear_layer.get_gradients();
    path = test_dir_path + "/self_weight_grads.npy";
    save_npy_matrix(gradients[0], path);
    path = test_dir_path + "/self_bias_grads.npy";
    save_npy_matrix(gradients[1], path);
    path = test_dir_path + "/neigh_weight_grads.npy";
    save_npy_matrix(gradients[2], path);
    path = test_dir_path + "/neigh_bias_grads.npy";
    save_npy_matrix(gradients[3], path);

    // compare with Pytorch, Numpy, SciPy
    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/integration_subset.py";
    system(command);

    // CLEAN-UP
    // destroy cuda handles
    cuda_helper.destroy_handles();

    // free memory
    free(features.values);
    free(classes.values);

    return 1; // TODO
}


TEST_CASE("Integration test subset", "[integration][subset]") {
    CHECK(integration_test_subset());
}
