// Copyright 2020 Marcel Wagenländer

#include "adam.hpp"
#include "add.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "feature_aggregation.hpp"
#include "helper.hpp"
#include "log_softmax.hpp"
#include "loss.hpp"
#include "relu.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"
#include "sparse_computation.hpp"

#include "catch2/catch.hpp"

const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";


int integration_test() {
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
    long num_nodes = features.num_rows_;
    long num_classes = 7;

    // layers
    FeatureAggregation graph_convolution_layer(&cuda_helper, &adjacency, "mean", num_nodes, features.num_columns_);
    Add add(&cuda_helper, num_nodes, features.num_columns_);
    SageLinear sage_linear_layer(&cuda_helper, features.num_columns_, num_classes, num_nodes);
    Relu relu_layer(&cuda_helper, num_nodes, num_classes);
    LogSoftmax log_softmax_layer(&cuda_helper, num_nodes, num_classes);
    NLLLoss loss_layer(num_nodes, num_classes);

    // optimiser
    Adam adam(&cuda_helper, learning_rate, sage_linear_layer.get_parameters(), sage_linear_layer.get_gradients());

    // graph convolution
    Matrix<float> *graph_convolution_result = graph_convolution_layer.forward(&features);
    path = test_dir_path + "/graph_convolution_result.npy";
    save_npy_matrix(graph_convolution_result, path);

    // linear layer
    Matrix<float> *linear_result = sage_linear_layer.forward(&features, graph_convolution_result);
    path = test_dir_path + "/linear_result.npy";
    save_npy_matrix(linear_result, path);
    std::vector<Matrix<float> *> parameters = sage_linear_layer.get_parameters();
    path = test_dir_path + "/self_weight.npy";
    save_npy_matrix(parameters[0], path);
    path = test_dir_path + "/self_bias.npy";
    save_npy_matrix(parameters[1], path);
    path = test_dir_path + "/neigh_weight.npy";
    save_npy_matrix(parameters[2], path);
    path = test_dir_path + "/neigh_bias.npy";
    save_npy_matrix(parameters[3], path);

    // ReLU
    Matrix<float> *relu_result = relu_layer.forward(linear_result);
    path = test_dir_path + "/relu_result.npy";
    save_npy_matrix(relu_result, path);

    // log-softmax
    Matrix<float> *log_softmax_result = log_softmax_layer.forward(relu_result);
    path = test_dir_path + "/log_softmax_result.npy";
    save_npy_matrix(log_softmax_result, path);

    // loss
    float loss = loss_layer.forward(log_softmax_result, &classes);
    Matrix<float> loss_mat(1, 1, true);
    loss_mat.values_[0] = loss;
    path = test_dir_path + "/loss_result.npy";
    save_npy_matrix(&loss_mat, path);

    // BACKPROPAGATION
    //loss
    Matrix<float> *loss_grads = loss_layer.backward();
    path = test_dir_path + "/loss_grads.npy";
    save_npy_matrix(loss_grads, path);

    // log-softmax
    Matrix<float> *log_softmax_grads = log_softmax_layer.backward(loss_grads);
    path = test_dir_path + "/log_softmax_grads.npy";
    save_npy_matrix(log_softmax_grads, path);

    // ReLU
    Matrix<float> *relu_grads = relu_layer.backward(log_softmax_grads);
    path = test_dir_path + "/relu_grads.npy";
    save_npy_matrix(relu_grads, path);

    // linear layer
    SageLinearGradients *linear_grads = sage_linear_layer.backward(relu_grads);
    path = test_dir_path + "/self_grads.npy";
    save_npy_matrix(linear_grads->self_gradients, path);
    path = test_dir_path + "/neigh_grads.npy";
    save_npy_matrix(linear_grads->neighbourhood_gradients, path);
    std::vector<Matrix<float> *> gradients = sage_linear_layer.get_gradients();
    path = test_dir_path + "/self_weight_grads.npy";
    save_npy_matrix(gradients[0], path);
    path = test_dir_path + "/self_bias_grads.npy";
    save_npy_matrix(gradients[1], path);
    path = test_dir_path + "/neigh_weight_grads.npy";
    save_npy_matrix(gradients[2], path);
    path = test_dir_path + "/neigh_bias_grads.npy";
    save_npy_matrix(gradients[3], path);

    // graph convolution
    Matrix<float> *graph_convolution_grads = graph_convolution_layer.backward(linear_grads->neighbourhood_gradients);
    path = test_dir_path + "/graph_convolution_grads.npy";
    save_npy_matrix(graph_convolution_grads, path);

    // add sage_linear_gradients.self_grads + gradients
    Matrix<float> *add_grads = add.forward(linear_grads->self_gradients, graph_convolution_grads);
    path = test_dir_path + "/add_grads.npy";
    save_npy_matrix(add_grads, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/integration.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int integration_test_chunked(long chunk_size) {
    // read features
    std::string path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    long num_nodes = features.num_rows_;
    long num_features = features.num_columns_;

    // read classes
    path = flickr_dir_path + "/classes.npy";
    Matrix<int> classes = load_npy_matrix<int>(path);

    // read adjacency
    path = flickr_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);

    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);

    std::vector<SparseMatrix<float>> adjacencies(num_chunks * num_chunks);
    double_chunk_up_sp(&adjacency, &adjacencies, chunk_size);
    Matrix<float> adjacency_row_sum(num_nodes, 1, true);
    sp_mat_sum_rows(&adjacency, &adjacency_row_sum);

    // FORWARD PASS
    CudaHelper cuda_helper;
    float learning_rate = 0.003;
    int num_classes = 7;

    // layers
    FeatureAggregationChunked feature_aggregation_layer(&cuda_helper, &adjacencies, &adjacency_row_sum, "mean", num_features, chunk_size, num_nodes);
    AddChunked add(&cuda_helper, chunk_size, num_nodes, features.num_columns_);
    SageLinearChunked sage_linear_layer(&cuda_helper, features.num_columns_, num_classes, chunk_size, num_nodes);
    ReluChunked relu_layer(&cuda_helper, chunk_size, num_nodes, num_classes);
    LogSoftmaxChunked log_softmax_layer(&cuda_helper, chunk_size, num_nodes, num_classes);
    NLLLoss loss_layer(num_nodes, num_classes);

    // optimiser
    Adam adam(&cuda_helper, learning_rate, sage_linear_layer.get_parameters(), sage_linear_layer.get_gradients());

    // graph convolution
    std::vector<Matrix<float>> *graph_convolution_result = feature_aggregation_layer.forward(&features_chunked);

    Matrix<float> graph_convolution_result_one(num_nodes, graph_convolution_result->at(0).num_columns_, false);
    stitch(graph_convolution_result, &graph_convolution_result_one);
    path = test_dir_path + "/graph_convolution_result.npy";
    save_npy_matrix(&graph_convolution_result_one, path);

    // linear layer
    std::vector<Matrix<float>> *linear_result = sage_linear_layer.forward(&features_chunked, graph_convolution_result);

    Matrix<float> linear_result_one(num_nodes, linear_result->at(0).num_columns_, false);
    stitch(linear_result, &linear_result_one);
    path = test_dir_path + "/linear_result.npy";
    save_npy_matrix(&linear_result_one, path);
    std::vector<Matrix<float> *> parameters = sage_linear_layer.get_parameters();
    path = test_dir_path + "/self_weight.npy";
    save_npy_matrix(parameters[0], path);
    path = test_dir_path + "/self_bias.npy";
    save_npy_matrix(parameters[1], path);
    path = test_dir_path + "/neigh_weight.npy";
    save_npy_matrix(parameters[2], path);
    path = test_dir_path + "/neigh_bias.npy";
    save_npy_matrix(parameters[3], path);

    // ReLU
    std::vector<Matrix<float>> *relu_result = relu_layer.forward(linear_result);

    Matrix<float> relu_result_one(num_nodes, relu_result->at(0).num_columns_, false);
    stitch(relu_result, &relu_result_one);
    path = test_dir_path + "/relu_result.npy";
    save_npy_matrix(&relu_result_one, path);

    // log-softmax
    std::vector<Matrix<float>> *log_softmax_result = log_softmax_layer.forward(relu_result);

    Matrix<float> log_softmax_result_one(num_nodes, log_softmax_result->at(0).num_columns_, false);
    stitch(log_softmax_result, &log_softmax_result_one);
    path = test_dir_path + "/log_softmax_result.npy";
    save_npy_matrix(&log_softmax_result_one, path);

    // loss
    float loss = loss_layer.forward(log_softmax_result, &classes);

    Matrix<float> loss_mat(1, 1, true);
    loss_mat.values_[0] = loss;
    path = test_dir_path + "/loss_result.npy";
    save_npy_matrix(&loss_mat, path);

    // BACKPROPAGATION
    //loss
    Matrix<float> *loss_grads = loss_layer.backward();
    path = test_dir_path + "/loss_grads.npy";
    save_npy_matrix(loss_grads, path);

    std::vector<Matrix<float>> loss_grads_chunked(num_chunks);
    chunk_up(loss_grads, &loss_grads_chunked, chunk_size);

    // log-softmax
    std::vector<Matrix<float>> *log_softmax_grads = log_softmax_layer.backward(&loss_grads_chunked);

    Matrix<float> log_softmax_grads_one(num_nodes, log_softmax_grads->at(0).num_columns_, false);
    stitch(log_softmax_grads, &log_softmax_grads_one);
    path = test_dir_path + "/log_softmax_grads.npy";
    save_npy_matrix(&log_softmax_grads_one, path);

    // ReLU
    std::vector<Matrix<float>> *relu_grads = relu_layer.backward(log_softmax_grads);

    Matrix<float> relu_grads_one(num_nodes, relu_grads->at(0).num_columns_, false);
    stitch(relu_grads, &relu_grads_one);
    path = test_dir_path + "/relu_grads.npy";
    save_npy_matrix(&relu_grads_one, path);

    // linear layer
    SageLinearGradientsChunked *linear_grads = sage_linear_layer.backward(relu_grads);

    Matrix<float> self_gradients_one(num_nodes, linear_grads->self_gradients->at(0).num_columns_, false);
    stitch(linear_grads->self_gradients, &self_gradients_one);
    path = test_dir_path + "/self_grads.npy";
    save_npy_matrix(&self_gradients_one, path);
    Matrix<float> neighbourhood_gradients_one(num_nodes, linear_grads->neighbourhood_gradients->at(0).num_columns_, false);
    stitch(linear_grads->neighbourhood_gradients, &neighbourhood_gradients_one);
    path = test_dir_path + "/neigh_grads.npy";
    save_npy_matrix(&neighbourhood_gradients_one, path);
    std::vector<Matrix<float> *> gradients = sage_linear_layer.get_gradients();
    path = test_dir_path + "/self_weight_grads.npy";
    save_npy_matrix(gradients[0], path);
    path = test_dir_path + "/self_bias_grads.npy";
    save_npy_matrix(gradients[1], path);
    path = test_dir_path + "/neigh_weight_grads.npy";
    save_npy_matrix(gradients[2], path);
    path = test_dir_path + "/neigh_bias_grads.npy";
    save_npy_matrix(gradients[3], path);

    // graph convolution
    std::vector<Matrix<float>> *graph_convolution_grads = feature_aggregation_layer.backward(linear_grads->neighbourhood_gradients);

    Matrix<float> graph_convolution_grads_one(num_nodes, graph_convolution_grads->at(0).num_columns_, false);
    stitch(graph_convolution_grads, &graph_convolution_grads_one);
    path = test_dir_path + "/graph_convolution_grads.npy";
    save_npy_matrix(&graph_convolution_grads_one, path);

    // add sage_linear_gradients.self_grads + gradients
    std::vector<Matrix<float>> *add_grads = add.forward(linear_grads->self_gradients, graph_convolution_grads);

    Matrix<float> add_grads_one(num_nodes, add_grads->at(0).num_columns_, false);
    stitch(add_grads, &add_grads_one);
    path = test_dir_path + "/add_grads.npy";
    save_npy_matrix(&add_grads_one, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/integration.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

TEST_CASE("Integration test", "[integration]") {
    CHECK(integration_test());
}

TEST_CASE("Integration test, chunked", "[integration][chunked]") {
    CHECK(integration_test_chunked(1 << 15));
    CHECK(integration_test_chunked(1 << 12));
    CHECK(integration_test_chunked(1 << 8));
}
