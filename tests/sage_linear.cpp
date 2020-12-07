// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
#include "cuda_helper.hpp"
#include "helper.hpp"
#include "tensors.hpp"
#include "chunking.hpp"

#include "catch2/catch.hpp"
#include <iostream>

const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";


int test_sage_linear(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *in_gradients) {
    std::string path;

    CudaHelper cuda_helper;
    SageLinear sage_linear_layer(&cuda_helper, input_self->num_columns_, in_gradients->num_columns_, input_self->num_rows_);

    Matrix<float> *result = sage_linear_layer.forward(input_self, input_neigh);

    SageLinearGradients *gradients = sage_linear_layer.backward(in_gradients);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);
    save_params(sage_linear_layer.get_parameters());
    save_grads(gradients, sage_linear_layer.get_gradients());

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_sage_linear_chunked(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *incoming_gradients, long chunk_size) {
    std::string path;
    CudaHelper cuda_helper;
    long num_nodes = input_self->num_rows_;
    long num_in_features = input_self->num_columns_;
    long num_out_features = incoming_gradients->num_columns_;

    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    std::vector<Matrix<float>> input_self_chunked(num_chunks);
    std::vector<Matrix<float>> input_neigh_chunked(num_chunks);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    chunk_up(input_self, &input_self_chunked, chunk_size);
    chunk_up(input_neigh, &input_neigh_chunked, chunk_size);
    chunk_up(incoming_gradients, &incoming_gradients_chunked, chunk_size);

    SageLinearChunked sage_linear_layer(&cuda_helper, num_in_features, num_out_features, chunk_size, num_nodes);

    std::vector<Matrix<float>> *activations = sage_linear_layer.forward(&input_self_chunked, &input_neigh_chunked);

    SageLinearGradientsChunked *gradients = sage_linear_layer.backward(&incoming_gradients_chunked);

    Matrix<float> activations_one(num_nodes, num_out_features, false);
    stitch(activations, &activations_one);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(&activations_one, path);
    save_params(sage_linear_layer.get_parameters());
    save_grads(gradients, sage_linear_layer.get_gradients(), num_nodes);

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

int test_sage_linear_nans(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *in_gradients) {
    std::string path;

    CudaHelper cuda_helper;
    SageLinear sage_linear_layer(&cuda_helper, input_self->num_columns_, in_gradients->num_columns_, input_self->num_rows_);

    Matrix<float> *result = sage_linear_layer.forward(input_self, input_neigh);

    bool first_check = check_nans(result, "SageLinear forward");

    SageLinearGradients *gradients = sage_linear_layer.backward(in_gradients);

    bool second_check = check_nans(gradients->self_gradients, "SageLinear backward self");
    bool third_check = check_nans(gradients->neighbourhood_gradients, "SageLinear backward neigh");

    if (first_check || second_check || third_check) {
        return 0;
    } else {
        return 1;
    }
}

int test_sage_linear_nans_chunked(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *incoming_gradients, long chunk_size) {
    std::string path;
    CudaHelper cuda_helper;
    long num_nodes = input_self->num_rows_;
    long num_in_features = input_self->num_columns_;
    long num_out_features = incoming_gradients->num_columns_;

    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    std::vector<Matrix<float>> input_self_chunked(num_chunks);
    std::vector<Matrix<float>> input_neigh_chunked(num_chunks);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    chunk_up(input_self, &input_self_chunked, chunk_size);
    chunk_up(input_neigh, &input_neigh_chunked, chunk_size);
    chunk_up(incoming_gradients, &incoming_gradients_chunked, chunk_size);

    SageLinearChunked sage_linear_layer(&cuda_helper, num_in_features, num_out_features, chunk_size, num_nodes);

    std::vector<Matrix<float>> *result = sage_linear_layer.forward(&input_self_chunked, &input_neigh_chunked);

    bool first_check = check_nans(result, "SageLinear forward");

    SageLinearGradientsChunked *gradients = sage_linear_layer.backward(&incoming_gradients_chunked);

    bool second_check = check_nans(gradients->self_gradients, "SageLinear backward self");
    bool third_check = check_nans(gradients->neighbourhood_gradients, "SageLinear backward neigh");

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

    CHECK(test_sage_linear(&input_self, &input_neigh, &in_gradients));
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

    CHECK(test_sage_linear_chunked(&input_self, &input_neigh, &in_gradients, 1 << 16));
    CHECK(test_sage_linear_chunked(&input_self, &input_neigh, &in_gradients, 1 << 12));
    CHECK(test_sage_linear_chunked(&input_self, &input_neigh, &in_gradients, 1 << 8));
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

    CHECK(test_sage_linear_nans_chunked(&input_self, &input_neigh, &in_gradients, 1 << 16));
    CHECK(test_sage_linear_nans_chunked(&input_self, &input_neigh, &in_gradients, 1 << 12));
    CHECK(test_sage_linear_nans_chunked(&input_self, &input_neigh, &in_gradients, 1 << 8));
}
