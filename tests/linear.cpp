// Copyright 2020 Marcel Wagenländer

#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "helper.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>

const std::string home = std::getenv("HOME");
const std::string flickr_dir_path =  "/mnt/data/flickr";
const std::string test_dir_path = home + "/gpu_memory_reduction/alzheimer/data/tests";


int test_linear() {
    std::string path;

    int rows = 1024;
    int num_in_features = 512;
    int num_out_features = 256;

    Matrix<float> input(rows, num_in_features, true);
    input.set_random_values();
    path = test_dir_path + "/input.npy";
    save_npy_matrix(&input, path);

    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_random_values();
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&in_gradients, path);

    CudaHelper cuda_helper;
    Linear linear(&cuda_helper, num_in_features, num_out_features, rows);

    Matrix<float> *activations = linear.forward(&input);
    Matrix<float> *input_gradients = linear.backward(&in_gradients);

    path = test_dir_path + "/activations.npy";
    save_npy_matrix(activations, path);
    path = test_dir_path + "/input_gradients.npy";
    save_npy_matrix(input_gradients, path);

    std::vector<Matrix<float> *> params = linear.get_parameters();
    std::vector<Matrix<float> *> weight_gradients = linear.get_gradients();
    path = test_dir_path + "/weight.npy";
    save_npy_matrix(params[0], path);
    path = test_dir_path + "/bias.npy";
    save_npy_matrix(params[1], path);
    path = test_dir_path + "/weight_gradients.npy";
    save_npy_matrix(weight_gradients[0], path);
    path = test_dir_path + "/bias_gradients.npy";
    save_npy_matrix(weight_gradients[1], path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/linear.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_linear_chunked(LinearChunked *linear, long chunk_size, bool keep_allocation) {
    std::string path;

    int rows = 1 << 17;
    int num_in_features = 512;
    int num_out_features = 256;

    Matrix<float> input(rows, num_in_features, true);
    input.set_random_values();
    path = test_dir_path + "/input.npy";
    save_npy_matrix(&input, path);

    Matrix<float> incoming_gradients(rows, num_out_features, true);
    incoming_gradients.set_random_values();
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&incoming_gradients, path);

    long num_chunks = ceil((float) rows / (float) chunk_size);
    std::vector<Matrix<float>> input_chunked(num_chunks);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    chunk_up(&input, &input_chunked, chunk_size);
    chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);

    CudaHelper cuda_helper;
    linear->set(&cuda_helper, chunk_size, rows, num_in_features, num_out_features, keep_allocation);

    std::vector<Matrix<float>> *output = linear->forward(&input_chunked);
    std::vector<Matrix<float>> *input_gradients = linear->backward(&incoming_gradients_chunked);

    Matrix<float> output_one(rows, num_out_features, false);
    stitch(output, &output_one);
    path = test_dir_path + "/activations.npy";
    save_npy_matrix(&output_one, path);

    Matrix<float> input_gradients_one(rows, num_in_features, false);
    stitch(input_gradients, &input_gradients_one);
    path = test_dir_path + "/input_gradients.npy";
    save_npy_matrix(&input_gradients_one, path);

    std::vector<Matrix<float> *> params = linear->get_parameters();
    std::vector<Matrix<float> *> weight_gradients = linear->get_gradients();
    path = test_dir_path + "/weight.npy";
    save_npy_matrix(params[0], path);
    path = test_dir_path + "/bias.npy";
    save_npy_matrix(params[1], path);
    path = test_dir_path + "/weight_gradients.npy";
    save_npy_matrix(weight_gradients[0], path);
    path = test_dir_path + "/bias_gradients.npy";
    save_npy_matrix(weight_gradients[1], path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/linear.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

TEST_CASE("Linear", "[linear]") {
    CHECK(test_linear());
}

TEST_CASE("Linear, chunked", "[linear][chunked]") {
    LinearChunked linear;
    CHECK(test_linear_chunked(&linear, 1 << 16, false));
    CHECK(test_linear_chunked(&linear, 1 << 12, false));
    CHECK(test_linear_chunked(&linear, 1 << 8, false));
}

TEST_CASE("Linear, chunked, keep", "[linear][chunked][keep]") {
    LinearChunked linear;
    CHECK(test_linear_chunked(&linear, 1 << 16, true));
    CHECK(test_linear_chunked(&linear, 1 << 12, true));
    CHECK(test_linear_chunked(&linear, 1 << 8, true));
}

TEST_CASE("Linear, pipelined", "[linear][pipelined]") {
    LinearPipelined linear;
    CHECK(test_linear_chunked(&linear, 1 << 16, false));
    CHECK(test_linear_chunked(&linear, 1 << 12, false));
    CHECK(test_linear_chunked(&linear, 1 << 8, false));
}
