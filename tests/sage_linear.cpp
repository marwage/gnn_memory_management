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


int test_sage_linear(matrix<float> input_self, matrix<float> input_neigh, matrix<float> in_gradients, int chunk_size) {
    std::string path;

    CudaHelper cuda_helper;
    SageLinearParent *sage_linear_layer;
    if (chunk_size == 0) {// no chunking
        sage_linear_layer = new SageLinear(&cuda_helper, input_self.columns, in_gradients.columns, input_self.rows);
    } else {
        sage_linear_layer = new SageLinearChunked(&cuda_helper, input_self.columns, in_gradients.columns, chunk_size, input_self.rows);
    }

    matrix<float> result = sage_linear_layer->forward(input_self, input_neigh);
    SageLinearGradients gradients = sage_linear_layer->backward(in_gradients);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);

    save_params(sage_linear_layer->get_parameters());

    save_grads(&gradients, sage_linear_layer->get_gradients());

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

    matrix<float> input_self = gen_non_rand_matrix(rows, columns);
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(input_self, path);

    matrix<float> input_neigh = gen_non_rand_matrix(rows, columns);
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(input_neigh, path);

    matrix<float> in_gradients = gen_non_rand_matrix(rows, num_out_features);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CudaHelper cuda_helper;
    SageLinear sage_linear(&cuda_helper, columns, num_out_features, rows);

    matrix<float> result = sage_linear.forward(input_self, input_neigh);
    SageLinearGradients gradients = sage_linear.backward(in_gradients);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);

    save_params(sage_linear.get_parameters());

    save_grads(&gradients, sage_linear.get_gradients());

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_parameters() {
    CudaHelper cuda_helper;
    int num_in_features = 1024;
    int num_out_features = 512;
    int num_nodes = 2048;
    int chunk_size = 128;
    SageLinearChunked sage_linear_chunked(&cuda_helper, num_in_features, num_out_features, chunk_size, num_nodes);

    std::vector<SageLinear> sage_linear_layers = sage_linear_chunked.get_layers();
    matrix<float> *params = sage_linear_layers[0].get_parameters();

    int equal = 1;
    int num_chunks = ceil((float) num_nodes / (float) chunk_size);
    for (int i = 1; i < num_chunks; ++i) {
        matrix<float> *other_params = sage_linear_layers[i].get_parameters();
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < params[j].rows * params[j].columns; ++k) {
                if (params[j].values[k] != other_params[j].values[k]) {
                    equal = 0;
                }
            }
        }
    }

    return equal;
}

int compare_mat(matrix<float> mat_a, matrix<float> mat_b, std::string name) {
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

int compare_sage_linear(int chunk_size) {
    int works = 1;
    int return_value;

    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    matrix<float> input_self = gen_rand_matrix(rows, columns);
    matrix<float> input_neigh = gen_rand_matrix(rows, columns);
    matrix<float> in_gradients = gen_rand_matrix(rows, num_out_features);

    CudaHelper cuda_helper;
    int num_nodes = rows;
    SageLinear sage_linear(&cuda_helper, columns, num_out_features, rows);
    SageLinearChunked sage_linear_chunked(&cuda_helper, columns, num_out_features, chunk_size, num_nodes);
    sage_linear_chunked.set_parameters(sage_linear.get_parameters());

    matrix<float> activations = sage_linear.forward(input_self, input_neigh);
    matrix<float> activations_chunked = sage_linear_chunked.forward(input_self, input_neigh);
    works = works * compare_mat(activations, activations_chunked, "Activations");

    SageLinearGradients gradients = sage_linear_chunked.backward(in_gradients);
    SageLinearGradients gradients_chunked = sage_linear_chunked.backward(in_gradients);
    works = works * compare_mat(gradients.self_grads, gradients_chunked.self_grads, "Gradients self");
    works = works * compare_mat(gradients.neigh_grads, gradients_chunked.neigh_grads, "Gradients neighbourhood");

    matrix<float> *weight_gradients = sage_linear.get_gradients();
    matrix<float> *weight_gradients_chunked = sage_linear_chunked.get_gradients();
    works = works * compare_mat(weight_gradients[0], weight_gradients_chunked[0], "Gradients self weights");
    works = works * compare_mat(weight_gradients[1], weight_gradients_chunked[1], "Gradients self bias");
    works = works * compare_mat(weight_gradients[2], weight_gradients_chunked[2], "Gradients neighbourhood weights");
    works = works * compare_mat(weight_gradients[3], weight_gradients_chunked[3], "Gradients neighbourhood bias");

    return works;
}


TEST_CASE("SageLinear", "[sagelinear]") {
    std::string path;
    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    matrix<float> input_self = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(input_self, path);

    matrix<float> input_neigh = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(input_neigh, path);

    matrix<float> in_gradients = gen_rand_matrix(rows, num_out_features);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CHECK(test_sage_linear(input_self, input_neigh, in_gradients, 0));
}

TEST_CASE("SageLinear, non-random input", "[sagelinear][nonrandom]") {
    std::string path;
    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    matrix<float> input_self = gen_non_rand_matrix(rows, columns);
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(input_self, path);

    matrix<float> input_neigh = gen_non_rand_matrix(rows, columns);
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(input_neigh, path);

    matrix<float> in_gradients = gen_non_rand_matrix(rows, num_out_features);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CHECK(test_sage_linear(input_self, input_neigh, in_gradients, 0));
}

TEST_CASE("SageLinear, chunked", "[sagelinear][chunked]") {
    std::string path;
    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    matrix<float> input_self = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(input_self, path);

    matrix<float> input_neigh = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(input_neigh, path);

    matrix<float> in_gradients = gen_rand_matrix(rows, num_out_features);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CHECK(test_sage_linear(input_self, input_neigh, in_gradients, 1 << 15));
    CHECK(test_sage_linear(input_self, input_neigh, in_gradients, 1 << 12));
    CHECK(test_sage_linear(input_self, input_neigh, in_gradients, 1 << 8));
}

TEST_CASE("SageLinear, chunked, non-random input", "[sagelinear][chunked][nonrandom]") {
    std::string path;
    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    matrix<float> input_self = gen_non_rand_matrix(rows, columns);
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(input_self, path);

    matrix<float> input_neigh = gen_non_rand_matrix(rows, columns);
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(input_neigh, path);

    matrix<float> in_gradients = gen_non_rand_matrix(rows, num_out_features);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CHECK(test_sage_linear(input_self, input_neigh, in_gradients, 1 << 15));
    CHECK(test_sage_linear(input_self, input_neigh, in_gradients, 1 << 12));
    CHECK(test_sage_linear(input_self, input_neigh, in_gradients, 1 << 8));
}

TEST_CASE("SageLinear, compare parameters", "[sagelinear][chunked][compare][parameters]") {
    CHECK(test_parameters());
}

TEST_CASE("SageLinear, compare", "[sagelinear][chunked][compare]") {
    CHECK(compare_sage_linear(1024));
    CHECK(compare_sage_linear(512));
    CHECK(compare_sage_linear(128));
}
