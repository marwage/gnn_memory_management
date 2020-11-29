// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "adam.hpp"
#include "cuda_helper.hpp"
#include "graph_convolution.hpp"
#include "helper.hpp"
#include "loss.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"
#include "add.hpp"

#include "catch2/catch.hpp"


int integration_test(int chunk_size) {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";

    // read features
    std::string path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    // read classes
    path = flickr_dir_path + "/classes.npy";
    Matrix<int> classes = load_npy_matrix<int>(path);

    // read adjacency
    path = flickr_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);

    // FORWARD PASS
    CudaHelper cuda_helper;
    float learning_rate = 0.003;
    int num_hidden_channels = 128;
    int num_classes = 7;

    // layers
    GraphConvolution graph_convolution_layer(&cuda_helper, &adjacency, "mean", features.rows, features.columns);
    SageLinearParent *sage_linear_layer;
    ReluParent *relu_layer;
    LogSoftmaxParent *log_softmax_layer;
    Add add(&cuda_helper, features.rows, features.columns);
    if (chunk_size == 0) {// no chunking
        sage_linear_layer = new SageLinear(&cuda_helper, features.columns, num_hidden_channels, features.rows);
        relu_layer = new Relu(&cuda_helper, features.rows, num_hidden_channels);
        log_softmax_layer = new LogSoftmax(&cuda_helper, features.rows, num_hidden_channels);
    } else {
        sage_linear_layer = new SageLinearChunked(&cuda_helper, features.columns, num_hidden_channels, chunk_size, features.rows);
        relu_layer = new ReluChunked(&cuda_helper, chunk_size, features.rows, num_hidden_channels);
        log_softmax_layer = new LogSoftmaxChunked(&cuda_helper, chunk_size, features.rows, num_hidden_channels);
    }
    NLLLoss loss_layer(features.rows, num_hidden_channels);

    // optimiser
    Adam adam(&cuda_helper, learning_rate, sage_linear_layer->get_parameters(), 4);

    // graph convolution
    Matrix<float> *graph_convolution_result = graph_convolution_layer.forward(&features);
    path = test_dir_path + "/graph_convolution_result.npy";
    save_npy_matrix(graph_convolution_result, path);

    // linear layer
    Matrix<float> *linear_result = sage_linear_layer->forward(&features, graph_convolution_result);
    path = test_dir_path + "/linear_result.npy";
    save_npy_matrix(linear_result, path);
    Matrix<float> **parameters = sage_linear_layer->get_parameters();
    path = test_dir_path + "/self_weight.npy";
    save_npy_matrix(parameters[0], path);
    path = test_dir_path + "/self_bias.npy";
    save_npy_matrix(parameters[1], path);
    path = test_dir_path + "/neigh_weight.npy";
    save_npy_matrix(parameters[2], path);
    path = test_dir_path + "/neigh_bias.npy";
    save_npy_matrix(parameters[3], path);

    // ReLU
    Matrix<float> *relu_result = relu_layer->forward(linear_result);
    path = test_dir_path + "/relu_result.npy";
    save_npy_matrix(relu_result, path);

    // log-softmax
    Matrix<float> *log_softmax_result = log_softmax_layer->forward(relu_result);
    path = test_dir_path + "/log_softmax_result.npy";
    save_npy_matrix(log_softmax_result, path);

    // loss
    float loss = loss_layer.forward(log_softmax_result, &classes);
    Matrix<float> loss_mat;
    loss_mat.rows = 1;
    loss_mat.columns = 1;
    loss_mat.values = &loss;
    path = test_dir_path + "/loss_result.npy";
    save_npy_matrix(loss_mat, path);

    // BACKPROPAGATION
    //loss
    Matrix<float> *loss_grads = loss_layer.backward();
    path = test_dir_path + "/loss_grads.npy";
    save_npy_matrix(loss_grads, path);

    // log-softmax
    Matrix<float> *log_softmax_grads = log_softmax_layer->backward(loss_grads);
    path = test_dir_path + "/log_softmax_grads.npy";
    save_npy_matrix(log_softmax_grads, path);

    // ReLU
    Matrix<float> *relu_grads = relu_layer->backward(log_softmax_grads);
    path = test_dir_path + "/relu_grads.npy";
    save_npy_matrix(relu_grads, path);

    // linear layer
    SageLinearGradients *linear_grads = sage_linear_layer->backward(relu_grads);
    path = test_dir_path + "/self_grads.npy";
    save_npy_matrix(linear_grads->self_grads, path);
    path = test_dir_path + "/neigh_grads.npy";
    save_npy_matrix(linear_grads->neigh_grads, path);
    Matrix<float> **gradients = sage_linear_layer->get_gradients();
    path = test_dir_path + "/self_weight_grads.npy";
    save_npy_matrix(gradients[0], path);
    path = test_dir_path + "/self_bias_grads.npy";
    save_npy_matrix(gradients[1], path);
    path = test_dir_path + "/neigh_weight_grads.npy";
    save_npy_matrix(gradients[2], path);
    path = test_dir_path + "/neigh_bias_grads.npy";
    save_npy_matrix(gradients[3], path);

    // graph convolution
    Matrix<float> *graph_convolution_grads = graph_convolution_layer.backward(linear_grads->neigh_grads);
    path = test_dir_path + "/graph_convolution_grads.npy";
    save_npy_matrix(graph_convolution_grads, path);

    // add sage_linear_gradients.self_grads + gradients
    Matrix<float> *add_grads = add.forward(linear_grads->self_grads, graph_convolution_grads);
    path = test_dir_path + "/add_grads.npy";
    save_npy_matrix(add_grads, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/integration.py";
    system(command);

    // CLEAN-UP
    // destroy cuda handles
    cuda_helper.destroy_handles();

    // free memory
    free(features.values);
    free(classes.values);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}


TEST_CASE("Integration test", "[integration]") {
    CHECK(integration_test(0));
}

TEST_CASE("Integration test, chunked", "[integration][chunked]") {
    CHECK(integration_test(1 < 15));
    CHECK(integration_test(1 < 12));
    CHECK(integration_test(1 < 8));
}
