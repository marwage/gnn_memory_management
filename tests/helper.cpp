// Copyright Marcel Wagenländer 2020

#include "helper.hpp"
#include "chunking.hpp"

#include <cmath>
#include <iostream>
#include <string>


std::string home = std::getenv("HOME");
std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
std::string test_dir_path = dir_path + "/tests";

void save_params(std::vector<Matrix<float> *> parameters) {
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

void save_grads(SageLinearGradients *gradients, std::vector<Matrix<float> *> weight_gradients) {
    std::string path;

    path = test_dir_path + "/self_grads.npy";
    save_npy_matrix(gradients->self_gradients, path);
    path = test_dir_path + "/neigh_grads.npy";
    save_npy_matrix(gradients->neighbourhood_gradients, path);

    path = test_dir_path + "/self_weight_grads.npy";
    save_npy_matrix(weight_gradients[0], path);
    path = test_dir_path + "/self_bias_grads.npy";
    save_npy_matrix(weight_gradients[1], path);
    path = test_dir_path + "/neigh_weight_grads.npy";
    save_npy_matrix(weight_gradients[2], path);
    path = test_dir_path + "/neigh_bias_grads.npy";
    save_npy_matrix(weight_gradients[3], path);
}

void save_grads(SageLinearGradientsChunked *gradients, std::vector<Matrix<float> *> weight_gradients, long num_nodes) {
    Matrix<float> self_gradients_one(num_nodes, gradients->self_gradients->at(0).num_columns_, true);
    stitch(gradients->self_gradients, &self_gradients_one);

    Matrix<float> neighbourhood_gradients_one(num_nodes, gradients->neighbourhood_gradients->at(0).num_columns_, true);
    stitch(gradients->neighbourhood_gradients, &neighbourhood_gradients_one);

    SageLinearGradients gradients_stitched;
    gradients_stitched.self_gradients = &self_gradients_one;
    gradients_stitched.neighbourhood_gradients = &neighbourhood_gradients_one;
    save_grads(&gradients_stitched, weight_gradients);
}

int read_return_value(std::string path) {
    Matrix<int> return_mat = load_npy_matrix<int>(path);
    return return_mat.values_[0];
}

void write_value(int value, std::string path) {
    Matrix<int> mat;
    mat.num_rows_ = 1;
    mat.num_columns_ = 1;
    mat.is_row_major_ = true;
    mat.values_ = new int[1];
    mat.values_[0] = value;
    save_npy_matrix(&mat, path);
}

int num_equal_rows(Matrix<float> A, Matrix<float> B) {
    int num_rows = 0;
    bool equal_row = true;

    for (int i = 0; i < A.num_rows_; ++i) {
        equal_row = true;
        for (int j = 0; j < A.num_columns_; ++j) {
            if (A.values_[j * A.num_rows_ + i] != B.values_[j * A.num_rows_ + i]) {
                equal_row = false;
            }
        }
        if (equal_row) {
            num_rows = num_rows + 1;
        }
    }

    return num_rows;
}

int compare_mat(Matrix<float> *mat_a, Matrix<float> *mat_b, std::string name) {
    std::string path_a = test_dir_path + "/a.npy";
    std::string path_b = test_dir_path + "/b.npy";
    std::string return_value_path = test_dir_path + "/value.npy";
    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/compare.py";

    save_npy_matrix(mat_a, path_a);
    save_npy_matrix(mat_b, path_b);
    system(command);
    int return_value = read_return_value(return_value_path);
    std::cout << name << ": " << return_value << std::endl;
    return return_value;
}
