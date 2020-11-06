// Copyright 2020 Marcel Wagenländer

#include "adam.hpp"
#include "sage_linear.hpp"
#include "cuda_helper.hpp"


int main() {
    int rows = 1024;
    int columns = 512;

    matrix<float> input_self;
    input_self.rows = rows;
    input_self.columns = columns;
    input_self.values = reinterpret_cast<float *>(
            malloc(input_self.rows * input_self.columns * sizeof(float)));
    for (int i = 0; i < input_self.rows * input_self.columns; ++i) {
        input_self.values[i] = rand();
    }

    matrix<float> input_neigh;
    input_neigh.rows = rows;
    input_neigh.columns = columns;
    input_neigh.values = reinterpret_cast<float *>(
            malloc(input_neigh.rows * input_neigh.columns * sizeof(float)));
    for (int i = 0; i < input_neigh.rows * input_neigh.columns; ++i) {
        input_neigh.values[i] = rand();
    }

    matrix<float> in_gradients;
    in_gradients.rows = rows;
    in_gradients.columns = columns;
    in_gradients.values = reinterpret_cast<float *>(
            malloc(in_gradients.rows * in_gradients.columns * sizeof(float)));
    for (int i = 0; i < in_gradients.rows * in_gradients.columns; ++i) {
        in_gradients.values[i] = rand();
    }

    CudaHelper cuda_helper;
    int num_out_features = 256;
    SageLinear sage_linear(columns, num_out_features, &cuda_helper);
    int chunk_size = 128;
    SageLinearChunked sage_linear_chunked(&cuda_helper, columns, num_out_features, chunk_size, rows);
    float learning_rate = 0.003;
    Adam adam(&cuda_helper, learning_rate, sage_linear.get_parameters(), 4);
    Adam adam_chunked(&cuda_helper, learning_rate, sage_linear_chunked.get_parameters(), 4);

    sage_linear.forward(input_self, input_neigh);
    sage_linear_chunked.forward(input_self, input_neigh);

    SageLinear::SageLinearGradients gradients = sage_linear.backward(in_gradients);
    SageLinear::SageLinearGradients gradients_chunked = sage_linear.backward(in_gradients);

    matrix<float> *adam_gradients = adam.step(sage_linear.get_gradients());
    matrix<float> *adam_chunked_gradients = adam_chunked.step(sage_linear_chunked.get_gradients());

    sage_linear.update_weights(adam_gradients);
    sage_linear_chunked.update_weights(adam_chunked_gradients);

    matrix<float> *normal_gradients = sage_linear.get_gradients();
    matrix<float> *chunked_gradients = sage_linear_chunked.get_gradients();

    matrix<float> *normal_weights = sage_linear.get_parameters();
    matrix<float> *chunked_weights = sage_linear_chunked.get_parameters();

    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";

    std::string path;
    for (int i = 0; i < 4; ++i) {
        path = test_dir_path + "/gradient_" + std::to_string(i) + ".npy";
        save_npy_matrix(normal_gradients[i], path);
        path = test_dir_path + "/gradient_chunked_" + std::to_string(i) + ".npy";
        save_npy_matrix(chunked_gradients[i], path);

        path = test_dir_path + "/weight_" + std::to_string(i) + ".npy";
        save_npy_matrix(normal_weights[i], path);
        path = test_dir_path + "/weight_chunked_" + std::to_string(i) + ".npy";
        save_npy_matrix(chunked_weights[i], path);
    }

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/adam.py";
    system(command);
}
