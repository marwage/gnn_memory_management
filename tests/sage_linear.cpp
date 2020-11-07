// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <iostream>


void save_params(matrix<float> *parameters) {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    path = test_dir_path + "/self_weight.npy";
    save_npy_matrix(parameters[0], path);
    path = test_dir_path + "/self_bias.npy";
    save_npy_matrix(parameters[1], path);
    path = test_dir_path + "/neigh_weight.npy";
    save_npy_matrix(parameters[2], path);
    path = test_dir_path + "/neigh_bias.npy";
    save_npy_matrix(parameters[3], path);
}

void save_grads(SageLinear::SageLinearGradients *gradients, matrix<float> *weight_gradients) {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    path = test_dir_path + "/self_grads.npy";
    save_npy_matrix(gradients->self_grads, path);
    path = test_dir_path + "/neigh_grads.npy";
    save_npy_matrix(gradients->neigh_grads, path);

    path = test_dir_path + "/self_weight_grads.npy";
    save_npy_matrix(weight_gradients[0], path);
    path = test_dir_path + "/self_bias_grads.npy";
    save_npy_matrix(weight_gradients[1], path);
    path = test_dir_path + "/neigh_weight_grads.npy";
    save_npy_matrix(weight_gradients[2], path);
    path = test_dir_path + "/neigh_bias_grads.npy";
    save_npy_matrix(weight_gradients[3], path);
}


int main() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    int rows = 1024;
    int columns = 512;
    int num_out_features = 256;

    matrix<float> input_self;
    input_self.rows = rows;
    input_self.columns = columns;
    input_self.values = reinterpret_cast<float *>(
            malloc(input_self.rows * input_self.columns * sizeof(float)));
    for (int i = 0; i < input_self.rows * input_self.columns; ++i) {
        input_self.values[i] = rand();
    }

    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(input_self, path);

    matrix<float> input_neigh;
    input_neigh.rows = rows;
    input_neigh.columns = columns;
    input_neigh.values = reinterpret_cast<float *>(
            malloc(input_neigh.rows * input_neigh.columns * sizeof(float)));
    for (int i = 0; i < input_neigh.rows * input_neigh.columns; ++i) {
        input_neigh.values[i] = rand();
    }

    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(input_neigh, path);

    matrix<float> in_gradients;
    in_gradients.rows = rows;
    in_gradients.columns = num_out_features;
    in_gradients.values = reinterpret_cast<float *>(
            malloc(in_gradients.rows * in_gradients.columns * sizeof(float)));
    for (int i = 0; i < in_gradients.rows * in_gradients.columns; ++i) {
        in_gradients.values[i] = rand();
    }

    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CudaHelper cuda_helper;
    SageLinear sage_linear(columns, num_out_features, &cuda_helper);

    matrix<float> result = sage_linear.forward(input_self, input_neigh);
    SageLinear::SageLinearGradients gradients = sage_linear.backward(in_gradients);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);

    save_params(sage_linear.get_parameters());

    save_grads(&gradients, sage_linear.get_gradients());

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear.py";
    system(command);

    std::cout << "---------- chunked ----------" << std::endl;

    int chunk_size = 128;
    int num_nodes = rows;
    SageLinearChunked sage_linear_chunked(&cuda_helper, columns, num_out_features, chunk_size, num_nodes);

    result = sage_linear_chunked.forward(input_self, input_neigh);
    gradients = sage_linear_chunked.backward(in_gradients);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);

    save_params(sage_linear_chunked.get_parameters());

    save_grads(&gradients, sage_linear_chunked.get_gradients());

    char command_chunked[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear.py";
    system(command_chunked);
}
