// Copyright 2020 Marcel Wagenländer

#include "alzheimer.hpp"
#include "adam.hpp"
#include "add.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "dropout.hpp"
#include "feature_aggregation.hpp"
#include "log_softmax.hpp"
#include "loss.hpp"
#include "relu.hpp"
#include "sage_linear.hpp"
#include "sparse_computation.hpp"
#include "tensors.hpp"

#include <iostream>

const std::string dir_path = "/mnt/data";


void alzheimer(Dataset dataset) {
    // read tensors
    // set path to directory
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);

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
    long num_hidden_channels = 256;
    long num_classes = get_dataset_num_classes(dataset);

    // layers
    NLLLoss loss_layer(num_nodes, num_classes);
    Add add_1(&cuda_helper, num_nodes, num_hidden_channels);
    Add add_2(&cuda_helper, num_nodes, num_hidden_channels);
    Dropout dropout_0(&cuda_helper, num_nodes, features.num_columns_);
    FeatureAggregation graph_convolution_0(&cuda_helper, &adjacency, "mean", num_nodes, features.num_columns_);
    SageLinear linear_0(&cuda_helper, features.num_columns_, num_hidden_channels, num_nodes);
    Relu relu_0(&cuda_helper, num_nodes, num_hidden_channels);
    Dropout dropout_1(&cuda_helper, num_nodes, num_hidden_channels);
    FeatureAggregation graph_convolution_1(&cuda_helper, &adjacency, "mean", num_nodes, num_hidden_channels);
    SageLinear linear_1(&cuda_helper, num_hidden_channels, num_hidden_channels, num_nodes);
    Relu relu_1(&cuda_helper, num_nodes, num_hidden_channels);
    Dropout dropout_2(&cuda_helper, num_nodes, num_hidden_channels);
    FeatureAggregation graph_convolution_2(&cuda_helper, &adjacency, "mean", num_nodes, num_hidden_channels);
    SageLinear linear_2(&cuda_helper, num_hidden_channels, num_classes, num_nodes);
    LogSoftmax log_softmax(&cuda_helper, num_nodes, num_classes);

    // optimizer
    long num_parameters = 6;
    std::vector<Matrix<float> *> parameters(num_parameters);
    std::vector<Matrix<float> *> params = linear_0.get_parameters();
    parameters[0] = params[0];
    parameters[1] = params[1];
    params = linear_1.get_parameters();
    parameters[2] = params[0];
    parameters[3] = params[1];
    params = linear_2.get_parameters();
    parameters[4] = params[0];
    parameters[5] = params[1];
    std::vector<Matrix<float> *> parameter_gradients(num_parameters);
    std::vector<Matrix<float> *> grads = linear_0.get_gradients();
    parameter_gradients[0] = grads[0];
    parameter_gradients[1] = grads[1];
    grads = linear_1.get_gradients();
    parameter_gradients[2] = grads[0];
    parameter_gradients[3] = grads[1];
    grads = linear_2.get_gradients();
    parameter_gradients[4] = grads[0];
    parameter_gradients[5] = grads[1];
    Adam adam(&cuda_helper, learning_rate, parameters, parameter_gradients);

    Matrix<float> *signals;
    Matrix<float> *signals_dropout;
    Matrix<float> *gradients;
    SageLinearGradients *sage_linear_gradients;
    float loss;

    int num_epochs = 10;
    for (int i = 0; i < num_epochs; ++i) {

        // dropout 0
        signals_dropout = dropout_0.forward(&features);

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
        loss = loss_layer.forward(signals, &classes);
        std::cout << "loss " << loss << std::endl;

        // BACKPROPAGATION
        //loss
        gradients = loss_layer.backward();

        // log-softmax
        gradients = log_softmax.backward(gradients);

        // linear layer 2
        sage_linear_gradients = linear_2.backward(gradients);

        // graph convolution 2
        gradients = graph_convolution_2.backward(sage_linear_gradients->neighbourhood_gradients);

        // add sage_linear_gradients.self_grads + gradients
        gradients = add_2.forward(sage_linear_gradients->self_gradients, gradients);

        // dropout 2
        gradients = dropout_2.backward(gradients);

        // relu 1
        gradients = relu_1.backward(gradients);

        // linear layer 1
        sage_linear_gradients = linear_1.backward(gradients);

        // graph convolution 1
        gradients = graph_convolution_1.backward(gradients);

        // add sage_linear_gradients.self_grads + gradients
        gradients = add_1.forward(sage_linear_gradients->self_gradients, gradients);

        // dropout 1
        gradients = dropout_1.backward(gradients);

        // relu 0
        gradients = relu_0.backward(gradients);

        // linear layer 0
        sage_linear_gradients = linear_0.backward(gradients);

        // no need for graph conv 0 and dropout 0

        // optimiser
        adam.step();
    }// end training loop
}

void alzheimer_chunked(Dataset dataset, long chunk_size) {
    // read tensors
    // set path to directory
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);

    // read features
    std::string path = dataset_path + "/features.npy";
    Matrix<float> *features = new Matrix<float>();
    load_npy_matrix<float>(path, features);

    long num_nodes = features->num_rows_;
    long num_features = features->num_columns_;

    // chunk features
    long num_chunks = ceil((float) features->num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(features, &features_chunked, chunk_size);

    delete features;

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
    SparseMatrix<float> *adjacency = new SparseMatrix<float>();
    load_mtx_matrix<float>(path, adjacency);

    // chunk adjacency
    std::vector<SparseMatrix<float>> adjacencies(num_chunks * num_chunks);
    double_chunk_up_sp(adjacency, &adjacencies, chunk_size);

    // get sums of adjacency rows
    Matrix<float> adjacency_row_sum(num_nodes, 1, true);
    sp_mat_sum_rows(adjacency, &adjacency_row_sum);

    delete adjacency;

    // FORWARD PASS
    CudaHelper cuda_helper;
    float learning_rate = 0.0003;
    long num_hidden_channels = 256;
    long num_classes = get_dataset_num_classes(dataset);

    // layers
    NLLLoss loss_layer(num_nodes, num_classes);
    AddChunked add_1(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    AddChunked add_2(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    DropoutChunked dropout_0(&cuda_helper, chunk_size, num_nodes, num_features);
    FeatureAggregationChunked graph_convolution_0(&cuda_helper, &adjacencies, &adjacency_row_sum, "mean", num_features, chunk_size, num_nodes);
    SageLinearChunked linear_0(&cuda_helper, num_features, num_hidden_channels, chunk_size, num_nodes);
    ReluChunked relu_0(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    DropoutChunked dropout_1(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    FeatureAggregationChunked graph_convolution_1(&cuda_helper, &adjacencies, &adjacency_row_sum, "mean", num_hidden_channels, chunk_size, num_nodes);
    SageLinearChunked linear_1(&cuda_helper, num_hidden_channels, num_hidden_channels, chunk_size, num_nodes);
    ReluChunked relu_1(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    DropoutChunked dropout_2(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    FeatureAggregationChunked graph_convolution_2(&cuda_helper, &adjacencies, &adjacency_row_sum, "mean", num_hidden_channels, chunk_size, num_nodes);
    SageLinearChunked linear_2(&cuda_helper, num_hidden_channels, num_classes, chunk_size, num_nodes);
    LogSoftmaxChunked log_softmax(&cuda_helper, chunk_size, num_nodes, num_classes);

    // optimizer
    long num_parameters = 6;
    std::vector<Matrix<float> *> parameters(num_parameters);
    std::vector<Matrix<float> *> params = linear_0.get_parameters();
    parameters[0] = params[0];
    parameters[1] = params[1];
    params = linear_1.get_parameters();
    parameters[2] = params[0];
    parameters[3] = params[1];
    params = linear_2.get_parameters();
    parameters[4] = params[0];
    parameters[5] = params[1];
    std::vector<Matrix<float> *> parameter_gradients(num_parameters);
    std::vector<Matrix<float> *> grads = linear_0.get_gradients();
    parameter_gradients[0] = grads[0];
    parameter_gradients[1] = grads[1];
    grads = linear_1.get_gradients();
    parameter_gradients[2] = grads[0];
    parameter_gradients[3] = grads[1];
    grads = linear_2.get_gradients();
    parameter_gradients[4] = grads[0];
    parameter_gradients[5] = grads[1];
    Adam adam(&cuda_helper, learning_rate, parameters, parameter_gradients);

    std::vector<Matrix<float>> *signals;
    std::vector<Matrix<float>> *signals_dropout;
    std::vector<Matrix<float>> *gradients;
    SageLinearGradientsChunked *sage_linear_gradients;
    Matrix<float> *loss_gradients;
    std::vector<Matrix<float>> loss_gradients_chunked(num_chunks);
    float loss;

    int num_epochs = 10;
    for (int i = 0; i < num_epochs; ++i) {

        // dropout 0
        signals_dropout = dropout_0.forward(&features_chunked);

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
        loss = loss_layer.forward(signals, &classes);
        std::cout << "loss " << loss << std::endl;

        // BACKPROPAGATION
        //loss
        loss_gradients = loss_layer.backward();

        chunk_up(loss_gradients, &loss_gradients_chunked, chunk_size);

        // log-softmax
        gradients = log_softmax.backward(&loss_gradients_chunked);

        // linear layer 2
        sage_linear_gradients = linear_2.backward(gradients);

        // graph convolution 2
        gradients = graph_convolution_2.backward(sage_linear_gradients->neighbourhood_gradients);

        // add sage_linear_gradients.self_grads + gradients
        gradients = add_2.forward(sage_linear_gradients->self_gradients, gradients);

        // dropout 2
        gradients = dropout_2.backward(gradients);

        // relu 1
        gradients = relu_1.backward(gradients);

        // linear layer 1
        sage_linear_gradients = linear_1.backward(gradients);

        // graph convolution 1
        gradients = graph_convolution_1.backward(gradients);

        // add sage_linear_gradients.self_grads + gradients
        gradients = add_1.forward(sage_linear_gradients->self_gradients, gradients);

        // dropout 1
        gradients = dropout_1.backward(gradients);

        // relu 0
        gradients = relu_0.backward(gradients);

        // linear layer 0
        sage_linear_gradients = linear_0.backward(gradients);

        // no need for graph conv 0 and dropout 0

        // optimiser
        adam.step();
    }// end training loop
}

void alzheimer_pipelined(Dataset dataset, long chunk_size) {
    // read tensors
    // set path to directory
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);

    // read features
    std::string path = dataset_path + "/features.npy";
    Matrix<float> *features = new Matrix<float>();
    load_npy_matrix<float>(path, features);

    long num_nodes = features->num_rows_;
    long num_features = features->num_columns_;

    // chunk features
    long num_chunks = ceil((float) features->num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(features, &features_chunked, chunk_size);

    delete features;

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
    SparseMatrix<float> *adjacency = new SparseMatrix<float>();
    load_mtx_matrix<float>(path, adjacency);

    // chunk adjacency
    std::vector<SparseMatrix<float>> adjacencies(num_chunks * num_chunks);
    double_chunk_up_sp(adjacency, &adjacencies, chunk_size);

    // get sums of adjacency rows
    Matrix<float> adjacency_row_sum(num_nodes, 1, true);
    sp_mat_sum_rows(adjacency, &adjacency_row_sum);

    delete adjacency;

    // FORWARD PASS
    CudaHelper cuda_helper;
    float learning_rate = 0.0003;
    long num_hidden_channels = 256;
    long num_classes = get_dataset_num_classes(dataset);

    // layers
    NLLLoss loss_layer(num_nodes, num_classes);
    AddPipelined add_1(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    AddPipelined add_2(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    DropoutPipelined dropout_0(&cuda_helper, chunk_size, num_nodes, num_features);
    FeatureAggregationPipelined graph_convolution_0(&cuda_helper, &adjacencies, &adjacency_row_sum, "mean", num_features, chunk_size, num_nodes);
    SageLinearPipelined linear_0(&cuda_helper, num_features, num_hidden_channels, chunk_size, num_nodes);
    ReluPipelined relu_0(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    DropoutPipelined dropout_1(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    FeatureAggregationPipelined graph_convolution_1(&cuda_helper, &adjacencies, &adjacency_row_sum, "mean", num_hidden_channels, chunk_size, num_nodes);
    SageLinearPipelined linear_1(&cuda_helper, num_hidden_channels, num_hidden_channels, chunk_size, num_nodes);
    ReluPipelined relu_1(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    DropoutPipelined dropout_2(&cuda_helper, chunk_size, num_nodes, num_hidden_channels);
    FeatureAggregationPipelined graph_convolution_2(&cuda_helper, &adjacencies, &adjacency_row_sum, "mean", num_hidden_channels, chunk_size, num_nodes);
    SageLinearPipelined linear_2(&cuda_helper, num_hidden_channels, num_classes, chunk_size, num_nodes);
    LogSoftmaxPipelined log_softmax(&cuda_helper, chunk_size, num_nodes, num_classes);

    // optimizer
    long num_parameters = 6;
    std::vector<Matrix<float> *> parameters(num_parameters);
    std::vector<Matrix<float> *> params = linear_0.get_parameters();
    parameters[0] = params[0];
    parameters[1] = params[1];
    params = linear_1.get_parameters();
    parameters[2] = params[0];
    parameters[3] = params[1];
    params = linear_2.get_parameters();
    parameters[4] = params[0];
    parameters[5] = params[1];
    std::vector<Matrix<float> *> parameter_gradients(num_parameters);
    std::vector<Matrix<float> *> grads = linear_0.get_gradients();
    parameter_gradients[0] = grads[0];
    parameter_gradients[1] = grads[1];
    grads = linear_1.get_gradients();
    parameter_gradients[2] = grads[0];
    parameter_gradients[3] = grads[1];
    grads = linear_2.get_gradients();
    parameter_gradients[4] = grads[0];
    parameter_gradients[5] = grads[1];
    Adam adam(&cuda_helper, learning_rate, parameters, parameter_gradients);

    std::vector<Matrix<float>> *signals;
    std::vector<Matrix<float>> *signals_dropout;
    std::vector<Matrix<float>> *gradients;
    SageLinearGradientsChunked *sage_linear_gradients;
    Matrix<float> *loss_gradients;
    std::vector<Matrix<float>> loss_gradients_chunked(num_chunks);
    float loss;

    int num_epochs = 10;
    for (int i = 0; i < num_epochs; ++i) {

        // dropout 0
        signals_dropout = dropout_0.forward(&features_chunked);

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
        loss = loss_layer.forward(signals, &classes);
        std::cout << "loss " << loss << std::endl;

        // BACKPROPAGATION
        //loss
        loss_gradients = loss_layer.backward();

        chunk_up(loss_gradients, &loss_gradients_chunked, chunk_size);

        // log-softmax
        gradients = log_softmax.backward(&loss_gradients_chunked);

        // linear layer 2
        sage_linear_gradients = linear_2.backward(gradients);

        // graph convolution 2
        gradients = graph_convolution_2.backward(sage_linear_gradients->neighbourhood_gradients);

        // add sage_linear_gradients.self_grads + gradients
        gradients = add_2.forward(sage_linear_gradients->self_gradients, gradients);

        // dropout 2
        gradients = dropout_2.backward(gradients);

        // relu 1
        gradients = relu_1.backward(gradients);

        // linear layer 1
        sage_linear_gradients = linear_1.backward(gradients);

        // graph convolution 1
        gradients = graph_convolution_1.backward(gradients);

        // add sage_linear_gradients.self_grads + gradients
        gradients = add_1.forward(sage_linear_gradients->self_gradients, gradients);

        // dropout 1
        gradients = dropout_1.backward(gradients);

        // relu 0
        gradients = relu_0.backward(gradients);

        // linear layer 0
        sage_linear_gradients = linear_0.backward(gradients);

        // no need for graph conv 0 and dropout 0

        // optimiser
        adam.step();
    }// end training loop
}
