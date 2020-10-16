// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "cuda_helper.hpp"
#include "dropout.hpp"
#include "graph_convolution.hpp"
#include "loss.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"
#include <adam.hpp>


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
    float learning_rate = 0.003;
    int num_hidden_channels = 128;
    int num_classes = 7;

    // layers
    Dropout dropout_0(&cuda_helper);
    GraphConvolution graph_convolution_0(&cuda_helper, &adjacency, "mean");
    SageLinear linear_0(features.columns, num_hidden_channels, &cuda_helper);
    Relu relu_0(&cuda_helper);
    Dropout dropout_1(&cuda_helper);
    GraphConvolution graph_convolution_1(&cuda_helper, &adjacency, "mean");
    SageLinear linear_1(num_hidden_channels, num_hidden_channels, &cuda_helper);
    Relu relu_1(&cuda_helper);
    Dropout dropout_2(&cuda_helper);
    GraphConvolution graph_convolution_2(&cuda_helper, &adjacency, "mean");
    SageLinear linear_2(num_hidden_channels, num_classes, &cuda_helper);
    LogSoftmax log_softmax(&cuda_helper);
    NLLLoss loss_layer;

    // optimizer
    Adam adam_0(&cuda_helper, learning_rate, linear_0.get_parameters(), 4);
    Adam adam_1(&cuda_helper, learning_rate, linear_1.get_parameters(), 4);
    Adam adam_2(&cuda_helper, learning_rate, linear_2.get_parameters(), 4);

    matrix<float> signals;
    matrix<float> signals_dropout;
    matrix<float> gradients;
    SageLinear::SageLinearGradients sage_linear_gradients;
    float loss;

    int num_epochs = 10;
    for (int i = 0; i < num_epochs; ++i) {
        // dropout 0
        signals_dropout = dropout_0.forward(features);

        // graph convolution 0
        signals = graph_convolution_0.forward(signals_dropout);

        // linear layer 0
        signals = linear_0.forward(signals_dropout, signals);

        // ReLU 0
        signals = relu_0.forward(signals);

        // dropout 1
        signals_dropout = dropout_1.forward(signals);

        // graph convolution 1
        signals = graph_convolution_1.forward(signals_dropout);

        // linear layer 1
        signals = linear_1.forward(signals_dropout, signals);

        // ReLU 1
        signals = relu_1.forward(signals);

        // dropout 2
        signals_dropout = dropout_2.forward(signals);

        // graph convolution 2
        signals = graph_convolution_2.forward(signals_dropout);

        // linear layer 2
        signals = linear_2.forward(signals_dropout, signals);

        // log-softmax
        signals = log_softmax.forward(signals);

        // loss
        loss = loss_layer.forward(signals, classes);
        std::cout << "loss " << loss << std::endl;

        // BACKPROPAGATION
        //loss
        gradients = loss_layer.backward();

        // log-softmax
        gradients = log_softmax.backward(gradients);

        // linear layer 2
        sage_linear_gradients = linear_2.backward(gradients);

        // graph convolution 2
        gradients = graph_convolution_2.backward(sage_linear_gradients.neigh_grads);

        // add sage_linear_gradients.self_grads + gradients
        gradients = add_matrices(&cuda_helper, sage_linear_gradients.self_grads, gradients);

        // dropout 2
        gradients = dropout_2.backward(gradients);

        // relu 1
        gradients = relu_1.backward(gradients);

        // linear layer 1
        sage_linear_gradients = linear_1.backward(gradients);

        // graph convolution 1
        gradients = graph_convolution_1.backward(gradients);

        // add sage_linear_gradients.self_grads + gradients
        gradients = add_matrices(&cuda_helper, sage_linear_gradients.self_grads, gradients);

        // dropout 1
        gradients = dropout_1.backward(gradients);

        // relu 0
        gradients = relu_0.backward(gradients);

        // linear layer 0
        sage_linear_gradients = linear_0.backward(gradients);

        // no need for graph conv 0 and dropout 0

        // optimiser
        matrix<float> *gradient_0 = adam_0.step(linear_0.get_gradients());
        matrix<float> *gradient_1 = adam_1.step(linear_1.get_gradients());
        matrix<float> *gradient_2 = adam_2.step(linear_2.get_gradients());

        // update weights
        linear_0.update_weights(gradient_0);
        linear_1.update_weights(gradient_1);
        linear_2.update_weights(gradient_2);


    }// end training loop

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
