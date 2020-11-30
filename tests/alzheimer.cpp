// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "adam.hpp"
#include "add.hpp"
#include "cuda_helper.hpp"
#include "dropout.hpp"
#include "graph_convolution.hpp"
#include "helper.hpp"
#include "loss.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <iostream>


int test_alzheimer(std::string dataset, int chunk_size) {
    // read tensors
    // set path to directory
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string dataset_path = dir_path + "/" + dataset;

    // read features
    std::string path = dataset_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    // read classes
    path = dataset_path + "/classes.npy";
    Matrix<int> classes = load_npy_matrix<int>(path);

    //    // read train_mask
    //    path = dataset_path + "/train_mask.npy";
    //    matrix<bool> train_mask = load_npy_matrix<bool>(path);
    //
    //    // read val_mask
    //    path = dataset_path + "/val_mask.npy";
    //    matrix<bool> val_mask = load_npy_matrix<bool>(path);
    //
    //    // read test_mask
    //    path = dataset_path + "/test_mask.npy";
    //    matrix<bool> test_mask = load_npy_matrix<bool>(path);

    // read adjacency
    path = dataset_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);

    // FORWARD PASS
    CudaHelper cuda_helper;
    long num_nodes = features.num_rows_;
    float learning_rate = 0.0003;
    long num_hidden_channels = 128;
    long num_classes;
    if (dataset == "flickr") {
        num_classes = 7;
    } else if (dataset == "reddit") {
        num_classes = 41;
    }

    // layers
    DropoutParent *dropout_0;
    GraphConvolution graph_convolution_0(&cuda_helper, &adjacency, "mean", num_nodes, features.num_columns_);
    SageLinearParent *linear_0;
    ReluParent *relu_0;
    DropoutParent *dropout_1;
    GraphConvolution graph_convolution_1(&cuda_helper, &adjacency, "mean", num_nodes, num_hidden_channels);
    SageLinearParent *linear_1;
    ReluParent *relu_1;
    DropoutParent *dropout_2;
    GraphConvolution graph_convolution_2(&cuda_helper, &adjacency, "mean", num_nodes, num_hidden_channels);
    SageLinearParent *linear_2;
    LogSoftmaxParent *log_softmax;
    NLLLoss loss_layer(num_nodes, num_classes);
    Add add_1(&cuda_helper, num_nodes, num_hidden_channels);
    Add add_2(&cuda_helper, num_nodes, num_hidden_channels);
    if (chunk_size == 0) {// no chunking
        dropout_0 = new Dropout(&cuda_helper, num_nodes, features.num_columns_);
        linear_0 = new SageLinear(&cuda_helper, features.num_columns_, num_hidden_channels, num_nodes);
        relu_0 = new Relu(&cuda_helper, num_nodes, num_hidden_channels);
        dropout_1 = new Dropout(&cuda_helper, num_nodes, num_hidden_channels);
        linear_1 = new SageLinear(&cuda_helper, num_hidden_channels, num_hidden_channels, num_nodes);
        relu_1 = new Relu(&cuda_helper, num_nodes, num_hidden_channels);
        dropout_2 = new Dropout(&cuda_helper, num_nodes, num_hidden_channels);
        linear_2 = new SageLinear(&cuda_helper, num_hidden_channels, num_classes, num_nodes);
        log_softmax = new LogSoftmax(&cuda_helper, num_nodes, num_classes);
    } else {
        dropout_0 = new DropoutChunked(&cuda_helper, chunk_size, num_nodes, features.num_columns_);
        linear_0 = new SageLinearChunked(&cuda_helper, features.num_columns_, num_hidden_channels, chunk_size, num_nodes);
        relu_0 = new ReluChunked(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
        dropout_1 = new DropoutChunked(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
        linear_1 = new SageLinearChunked(&cuda_helper, num_hidden_channels, num_hidden_channels, chunk_size, num_nodes);
        relu_1 = new ReluChunked(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
        dropout_2 = new DropoutChunked(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
        linear_2 = new SageLinearChunked(&cuda_helper, num_hidden_channels, num_classes, chunk_size, num_nodes);
        log_softmax = new LogSoftmaxChunked(&cuda_helper, chunk_size, num_nodes, num_classes);
    }

    // optimizer
    Adam adam_0(&cuda_helper, learning_rate, linear_0->get_parameters(), 4);
    Adam adam_1(&cuda_helper, learning_rate, linear_1->get_parameters(), 4);
    Adam adam_2(&cuda_helper, learning_rate, linear_2->get_parameters(), 4);

    Matrix<float> *signals;
    Matrix<float> *signals_dropout;
    Matrix<float> *gradients;
    SageLinearGradients *sage_linear_gradients;
    Matrix<float> *gradient_0;
    Matrix<float> *gradient_1;
    Matrix<float> *gradient_2;
    float loss;

    int num_epochs = 10;
    for (int i = 0; i < num_epochs; ++i) {
        // DEBUGGING
        std::cout << "Iteration " << i << std::endl;

        // dropout 0
        signals_dropout = dropout_0->forward(&features);

        // DEBUGGING
        if (check_nans(signals_dropout, "Dropout 0")) return 0;

        // graph convolution 0
        signals = graph_convolution_0.forward(signals_dropout);

        // DEBUGGING
        if (check_nans(signals, "Graph convolution 0")) return 0;

        // linear layer 0
        signals = linear_0->forward(signals_dropout, signals);

        // DEBUGGING
        if (check_nans(signals, "Sage-linear 0")) return 0;

        // ReLU 0
        signals = relu_0->forward(signals);

        // DEBUGGING
        if (check_nans(signals, "ReLU 0")) return 0;

        // dropout 1
        signals_dropout = dropout_1->forward(signals);

        // DEBUGGING
        if (check_nans(signals_dropout, "Dropout 1")) return 0;

        // graph convolution 1
        signals = graph_convolution_1.forward(signals_dropout);

        // DEBUGGING
        if (check_nans(signals, "Graph convolution 1")) return 0;

        // linear layer 1
        signals = linear_1->forward(signals_dropout, signals);

        // DEBUGGING
        if (check_nans(signals, "Sage-Linear 1")) return 0;

        // ReLU 1
        signals = relu_1->forward(signals);

        // DEBUGGING
        if (check_nans(signals, "ReLU 1")) return 0;

        // dropout 2
        signals_dropout = dropout_2->forward(signals);

        // DEBUGGING
        if (check_nans(signals_dropout, "Dropout 2")) return 0;

        // graph convolution 2
        signals = graph_convolution_2.forward(signals_dropout);

        // DEBUGGING
        if (check_nans(signals, "Graph convolution 2")) return 0;

        // linear layer 2
        signals = linear_2->forward(signals_dropout, signals);

        // DEBUGGING
        if (check_nans(signals, "Sage-Linear 2")) return 0;

        // log-softmax
        signals = log_softmax->forward(signals);

        // DEBUGGING
        if (check_nans(signals, "Log-softmax 2")) return 0;

        // loss
        loss = loss_layer.forward(signals, &classes);
        std::cout << "loss " << loss << std::endl;

        // BACKPROPAGATION
        //loss
        gradients = loss_layer.backward();

        // DEBUGGING
        if (check_nans(gradients, "Loss gradients")) return 0;

        // log-softmax
        gradients = log_softmax->backward(gradients);

        // DEBUGGING
        if (check_nans(gradients, "Log-softmax gradients")) return 0;

        // linear layer 2
        sage_linear_gradients = linear_2->backward(gradients);

        // DEBUGGING
        if (check_nans(sage_linear_gradients->self_grads, "Sage-Linear 2 self gradients")) return 0;
        if (check_nans(sage_linear_gradients->neigh_grads, "Sage-Linear 2 neighbourhood gradients")) return 0;

        // graph convolution 2
        gradients = graph_convolution_2.backward(sage_linear_gradients->neigh_grads);

        // DEBUGGING
        if (check_nans(gradients, "Graph convolution 2 gradients")) return 0;

        // add sage_linear_gradients.self_grads + gradients
        gradients = add_2.forward(sage_linear_gradients->self_grads, gradients);

        // DEBUGGING
        if (check_nans(gradients, "Add 2 gradients")) return 0;

        // dropout 2
        gradients = dropout_2->backward(gradients);

        // DEBUGGING
        if (check_nans(gradients, "Dropout 2 gradients")) return 0;

        // relu 1
        gradients = relu_1->backward(gradients);

        // DEBUGGING
        if (check_nans(gradients, "ReLU 1 gradients")) return 0;

        // linear layer 1
        sage_linear_gradients = linear_1->backward(gradients);

        // DEBUGGING
        if (check_nans(sage_linear_gradients->self_grads, "Sage-Linear 1 self gradients")) return 0;
        if (check_nans(sage_linear_gradients->neigh_grads, "Sage-Linear 1 neighbourhood gradients")) return 0;

        // graph convolution 1
        gradients = graph_convolution_1.backward(gradients);

        // DEBUGGING
        if (check_nans(gradients, "Graph convolution 1 gradients")) return 0;

        // add sage_linear_gradients.self_grads + gradients
        gradients = add_1.forward(sage_linear_gradients->self_grads, gradients);

        // DEBUGGING
        if (check_nans(gradients, "Add 1 gradients")) return 0;

        // dropout 1
        gradients = dropout_1->backward(gradients);

        // DEBUGGING
        if (check_nans(gradients, "Dropout 1 gradients")) return 0;

        // relu 0
        gradients = relu_0->backward(gradients);

        // DEBUGGING
        if (check_nans(gradients, "ReLU 1 gradients")) return 0;

        // linear layer 0
        sage_linear_gradients = linear_0->backward(gradients);

        // DEBUGGING
        if (check_nans(sage_linear_gradients->self_grads, "Sage-Linear 0 self gradients")) return 0;
        if (check_nans(sage_linear_gradients->neigh_grads, "Sage-Linear 0 neighbourhood gradients")) return 0;

        // no need for graph conv 0 and dropout 0

        // optimiser
        linear_0->update_weights(adam_0.step(linear_0->get_gradients()));
        linear_1->update_weights(adam_1.step(linear_1->get_gradients()));
        linear_2->update_weights(adam_2.step(linear_2->get_gradients()));
    }// end training loop

    // CLEAN-UP
    // destroy cuda handles
    cuda_helper.destroy_handles();

    return 1;
}

TEST_CASE("Alzeheimer", "[alzheimer]") {
    CHECK(test_alzheimer("flickr", 0));
}
