// Copyright 2020 Marcel Wagenländer

#include "sage_linear_adam.hpp"
#include "cuda_helper.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"


matrix<float> gen_rand_matrix(int num_rows, int num_columns) {
    matrix<float> mat;
    mat.rows = num_rows;
    mat.columns = num_columns;
    mat.values = (float *) malloc(mat.rows * mat.columns * sizeof(float));
    for (int i = 0; i < mat.rows * mat.columns; ++i) {
        mat.values[i] = rand();
    }

    return mat;
}

int check_adam() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    int rows = 1024;
    int columns = 512;

    int num_out_features = 256;
    int chunk_size = 128;
    float learning_rate = 0.003;

    matrix<float> input_self = gen_rand_matrix(rows, columns);
    save_npy_matrix(input_self, test_dir_path + "/input_self.npy");

    matrix<float> input_neigh = gen_rand_matrix(rows, columns);
    save_npy_matrix(input_neigh, test_dir_path + "/input_neigh.npy");

    matrix<float> in_gradients = gen_rand_matrix(rows, num_out_features);
    save_npy_matrix(in_gradients, test_dir_path + "/in_gradients.npy");

    CudaHelper cuda_helper;
    SageLinear sage_linear(columns, num_out_features, &cuda_helper);
    Adam adam(&cuda_helper, learning_rate, sage_linear.get_parameters(), 4);

    matrix<float> *parameters = sage_linear.get_parameters();
    save_npy_matrix(parameters[0], test_dir_path + "/self_weight.npy");
    save_npy_matrix(parameters[1], test_dir_path + "/self_bias.npy");
    save_npy_matrix(parameters[2], test_dir_path + "/neigh_weight.npy");
    save_npy_matrix(parameters[3], test_dir_path + "/neigh_bias.npy");

    matrix<float> activations = sage_linear.forward(input_self, input_neigh);

    save_npy_matrix(activations, test_dir_path + "/activations.npy");

    SageLinear::SageLinearGradients gradients = sage_linear.backward(in_gradients);

    save_npy_matrix(gradients.self_grads, test_dir_path + "/self_grads.npy");
    save_npy_matrix(gradients.neigh_grads, test_dir_path + "/neigh_grads.npy");

    matrix<float> *weight_gradients = sage_linear.get_gradients();
    save_npy_matrix(weight_gradients[0], test_dir_path + "/self_weight_grads.npy");
    save_npy_matrix(weight_gradients[1], test_dir_path + "/self_bias_grads.npy");
    save_npy_matrix(weight_gradients[2], test_dir_path + "/neigh_weight_grads.npy");
    save_npy_matrix(weight_gradients[3], test_dir_path + "/neigh_bias_grads.npy");

    matrix<float> *adam_gradients = adam.step(sage_linear.get_gradients());
    sage_linear.update_weights(adam_gradients);

    matrix<float> *weights_updated = sage_linear.get_parameters();
    save_npy_matrix(weights_updated[0], test_dir_path + "/self_weight_updated.npy");
    save_npy_matrix(weights_updated[1], test_dir_path + "/self_bias_updated.npy");
    save_npy_matrix(weights_updated[2], test_dir_path + "/neigh_weight_updated.npy");
    save_npy_matrix(weights_updated[3], test_dir_path + "/neigh_bias_updated.npy");

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear_adam.py";
    system(command);
}

int main() {
    check_adam();
}
