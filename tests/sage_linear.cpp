// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
#include "cuda_helper.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <iostream>

const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";


int test_sage_linear(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *in_gradients, int chunk_size) {
    std::string path;

    CudaHelper cuda_helper;
    SageLinearParent *sage_linear_layer;
    if (chunk_size == 0) {// no chunking
        sage_linear_layer = new SageLinear(&cuda_helper, input_self->num_columns_, in_gradients->num_columns_, input_self->num_rows_);
    } else {
        sage_linear_layer = new SageLinearChunked(&cuda_helper, input_self->num_columns_, in_gradients->num_columns_, chunk_size, input_self->num_rows_);
    }

    Matrix<float> *result = sage_linear_layer->forward(input_self, input_neigh);

    SageLinearGradients *gradients = sage_linear_layer->backward(in_gradients);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);
    save_params(sage_linear_layer->get_parameters());
    save_grads(gradients, sage_linear_layer->get_gradients());

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_sage_linear_non_random(int chunk_size) {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    int rows = 1024;
    int columns = 512;
    int num_out_features = 256;

    Matrix<float> input_self(rows, columns, true);
    input_self.set_values(false);
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(&input_self, path);

    Matrix<float> input_neigh(rows, columns, true);
    input_neigh.set_values(false);
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(&input_neigh, path);

    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_values(false);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&in_gradients, path);

    CudaHelper cuda_helper;
    SageLinear sage_linear(&cuda_helper, columns, num_out_features, rows);

    Matrix<float> *result = sage_linear.forward(&input_self, &input_neigh);
    SageLinearGradients *gradients = sage_linear.backward(&in_gradients);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);

    save_params(sage_linear.get_parameters());

    save_grads(gradients, sage_linear.get_gradients());

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_sage_linear_nans(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *in_gradients, int chunk_size) {
    std::string path;

    CudaHelper cuda_helper;
    SageLinearParent *sage_linear_layer;
    if (chunk_size == 0) {// no chunking
        sage_linear_layer = new SageLinear(&cuda_helper, input_self->num_columns_, in_gradients->num_columns_, input_self->num_rows_);
    } else {
        sage_linear_layer = new SageLinearChunked(&cuda_helper, input_self->num_columns_, in_gradients->num_columns_, chunk_size, input_self->num_rows_);
    }

    Matrix<float> *result = sage_linear_layer->forward(input_self, input_neigh);

    bool first_check = check_nans(result, "SageLinear forward");

    SageLinearGradients *gradients = sage_linear_layer->backward(in_gradients);

    bool second_check = check_nans(gradients->self_grads, "SageLinear backward self");
    bool third_check = check_nans(gradients->neigh_grads, "SageLinear backward neigh");

    if (first_check || second_check || third_check) {
        return 0;
    } else {
        return 1;
    }
}

TEST_CASE("SageLinear", "[sagelinear]") {
    std::string path;
    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    Matrix<float> input_self(rows, columns, true);
    input_self.set_random_values();
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(&input_self, path);

    Matrix<float> input_neigh(rows, columns, true);
    input_neigh.set_random_values();
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(&input_neigh, path);

    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_random_values();
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&in_gradients, path);

    CHECK(test_sage_linear(&input_self, &input_neigh, &in_gradients, 0));
}

TEST_CASE("SageLinear, chunked", "[sagelinear][chunked]") {
    std::string path;
    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    Matrix<float> input_self(rows, columns, true);
    input_self.set_random_values();
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(&input_self, path);

    Matrix<float> input_neigh(rows, columns, true);
    input_neigh.set_random_values();
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(&input_neigh, path);

    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_random_values();
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&in_gradients, path);

    CHECK(test_sage_linear(&input_self, &input_neigh, &in_gradients, 1 << 16));
    CHECK(test_sage_linear(&input_self, &input_neigh, &in_gradients, 1 << 12));
    CHECK(test_sage_linear(&input_self, &input_neigh, &in_gradients, 1 << 8));
}

TEST_CASE("SageLinear, chunked, NaNs", "[sagelinear][chunked][nans]") {
    std::string path;
    int rows = 2449029; // products
    int columns = 100; // products
    int num_out_features = 1 << 8;

    Matrix<float> input_self(rows, columns, true);
    input_self.set_random_values();

    Matrix<float> input_neigh(rows, columns, true);
    input_neigh.set_random_values();

    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_random_values();

    CHECK(test_sage_linear_nans(&input_self, &input_neigh, &in_gradients, 1 << 16));
    CHECK(test_sage_linear_nans(&input_self, &input_neigh, &in_gradients, 1 << 12));
    CHECK(test_sage_linear_nans(&input_self, &input_neigh, &in_gradients, 1 << 8));
}
