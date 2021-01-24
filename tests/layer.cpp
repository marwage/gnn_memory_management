// Copyright 2020 Marcel Wagenländer

#include "layer.hpp"
#include "chunking.hpp"
#include "helper.hpp"
#include "tensors.hpp"
#include "dropout.hpp" // TEMPORARY

#include <string>

const std::string home = std::getenv("HOME");
const std::string flickr_dir_path = "/mnt/data/flickr";
const std::string test_dir_path = home + "/gpu_memory_reduction/alzheimer/data/tests";


int test_layer(Layer *layer, std::string py_name) {
    std::string path;
    CudaHelper cuda_helper;

    // input matrices
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> incoming_gradients(features.num_rows_, features.num_columns_, true);
    incoming_gradients.set_random_values();
    path = test_dir_path + "/incoming_gradients.npy";
    save_npy_matrix(&incoming_gradients, path);

    // layer
    layer->set(&cuda_helper, features.num_rows_, features.num_columns_);

    // forward
    Matrix<float> *output = layer->forward(&features);

    path = test_dir_path + "/output.npy";
    save_npy_matrix(output, path);

    // backward
    Matrix<float> *gradients = layer->backward(&incoming_gradients);

    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(gradients, path);

    // test against Pytorch
    std::string command = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/" + py_name + ".py";
    system(command.c_str());

    // read test result
    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_layer_chunked(LayerChunked *layer, std::string py_name, long chunk_size) {
    std::string path;
    CudaHelper cuda_helper;

    // inputs matrices
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> incoming_gradients(features.num_rows_, features.num_columns_, true);
    incoming_gradients.set_random_values();
    path = test_dir_path + "/incoming_gradients.npy";
    save_npy_matrix(&incoming_gradients, path);
    long num_nodes = features.num_rows_;
    long num_features = features.num_columns_;
    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);

    // layer
    layer->set(&cuda_helper, chunk_size, features.num_rows_, features.num_columns_);

    // forward
    std::vector<Matrix<float>> *output = layer->forward(&features_chunked);

    Matrix<float> output_one(num_nodes, num_features, false);
    stitch(output, &output_one);
    path = test_dir_path + "/output.npy";
    save_npy_matrix(&output_one, path);

    // backward
    std::vector<Matrix<float>> *gradients = layer->backward(&incoming_gradients_chunked);

    Matrix<float> gradients_one(num_nodes, num_features, false);
    stitch(gradients, &gradients_one);
    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(&gradients_one, path);

    // test against Pytorch
    std::string command = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/" + py_name + ".py";
    system(command.c_str());

    // read test result
    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

// TEMPORARY ONLY DROPOUT
int test_layer_chunked_keep(LayerChunked *layer, std::string py_name, long chunk_size) {
    std::string path;
    CudaHelper cuda_helper;

    // inputs matrices
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> incoming_gradients(features.num_rows_, features.num_columns_, true);
    incoming_gradients.set_random_values();
    path = test_dir_path + "/incoming_gradients.npy";
    save_npy_matrix(&incoming_gradients, path);
    long num_nodes = features.num_rows_;
    long num_features = features.num_columns_;
    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);

    // layer
    layer->set(&cuda_helper, chunk_size, features.num_rows_, features.num_columns_, true);

    // forward
    std::vector<Matrix<float>> *output = layer->forward(&features_chunked);

    Matrix<float> output_one(num_nodes, num_features, false);
    stitch(output, &output_one);
    path = test_dir_path + "/output.npy";
    save_npy_matrix(&output_one, path);

    // backward
    std::vector<Matrix<float>> *gradients = layer->backward(&incoming_gradients_chunked);

    Matrix<float> gradients_one(num_nodes, num_features, false);
    stitch(gradients, &gradients_one);
    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(&gradients_one, path);

    // test against Pytorch
    std::string command = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/" + py_name + ".py";
    system(command.c_str());

    // read test result
    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}
