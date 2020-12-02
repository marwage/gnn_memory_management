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

int test_parameters() {
    CudaHelper cuda_helper;
    int num_in_features = 1024;
    int num_out_features = 512;
    int num_nodes = 2048;
    int chunk_size = 128;
    SageLinearChunked sage_linear_chunked(&cuda_helper, num_in_features, num_out_features, chunk_size, num_nodes);

    std::vector<SageLinear> *sage_linear_layers = sage_linear_chunked.get_layers();
    Matrix<float> **params = sage_linear_layers->at(0).get_parameters();

    int equal = 1;
    long num_parameters = 4;
    int num_chunks = ceil((float) num_nodes / (float) chunk_size);
    for (int i = 1; i < num_chunks; ++i) {
        Matrix<float> **other_params = sage_linear_layers->at(0).get_parameters();
        for (int j = 0; j < num_parameters; ++j) {
            for (int k = 0; k < params[j]->num_rows_ * params[j]->num_columns_; ++k) {
                if (params[j]->values_[k] != other_params[j]->values_[k]) {
                    equal = 0;
                }
            }
        }
    }

    return equal;
}


int compare_sage_linear(int chunk_size) {
    int works = 1;

    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    Matrix<float> input_self(rows, columns, true);
    input_self.set_random_values();
    Matrix<float> input_neigh(rows, columns, true);
    input_neigh.set_random_values();
    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_random_values();

    CudaHelper cuda_helper;
    int num_nodes = rows;
    SageLinear sage_linear(&cuda_helper, columns, num_out_features, rows);
    SageLinearChunked sage_linear_chunked(&cuda_helper, columns, num_out_features, chunk_size, num_nodes);
    sage_linear_chunked.set_parameters(sage_linear.get_parameters());

    Matrix<float> *activations = sage_linear.forward(&input_self, &input_neigh);
    Matrix<float> *activations_chunked = sage_linear_chunked.forward(&input_self, &input_neigh);
    works = works * compare_mat(activations, activations_chunked, "Activations");

    SageLinearGradients *gradients = sage_linear_chunked.backward(&in_gradients);
    SageLinearGradients *gradients_chunked = sage_linear_chunked.backward(&in_gradients);
    works = works * compare_mat(gradients->self_grads, gradients_chunked->self_grads, "Gradients self");
    works = works * compare_mat(gradients->neigh_grads, gradients_chunked->neigh_grads, "Gradients neighbourhood");

    Matrix<float> **weight_gradients = sage_linear.get_gradients();
    Matrix<float> **weight_gradients_chunked = sage_linear_chunked.get_gradients();
    works = works * compare_mat(weight_gradients[0], weight_gradients_chunked[0], "Gradients self weights");
    works = works * compare_mat(weight_gradients[1], weight_gradients_chunked[1], "Gradients self bias");
    works = works * compare_mat(weight_gradients[2], weight_gradients_chunked[2], "Gradients neighbourhood weights");
    works = works * compare_mat(weight_gradients[3], weight_gradients_chunked[3], "Gradients neighbourhood bias");

    return works;
}

int test_sage_linear_set_parameters() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    long num_nodes = 2048;
    long num_in_features = 1024;
    long num_out_features = 512;
    long num_params = 4;

    Matrix<float> self_weight(num_in_features, num_out_features, false);
    self_weight.set_random_values();
    Matrix<float> self_bias(num_out_features, 1, false);
    self_bias.set_random_values();
    Matrix<float> neigh_weight(num_in_features, num_out_features, false);
    neigh_weight.set_random_values();
    Matrix<float> neigh_bias(num_out_features, 1, false);
    neigh_bias.set_random_values();

    CudaHelper cuda_helper;
    SageLinear sage_linear(&cuda_helper, num_in_features, num_out_features, num_nodes);

    Matrix<float> **parameters = new Matrix<float> *[num_params];
    parameters[0] = &self_weight;
    parameters[1] = &self_bias;
    parameters[2] = &neigh_weight;
    parameters[3] = &neigh_bias;

    sage_linear.set_parameters(parameters);
    Matrix<float> **get_parameters = sage_linear.get_parameters();

    compare_mat(parameters[0], get_parameters[0], "self weights");
    compare_mat(parameters[1], get_parameters[1], "self bias");
    compare_mat(parameters[2], get_parameters[2], "neigh weights");
    compare_mat(parameters[3], get_parameters[3], "neigh bias");

    return 1;// TODO
}

int test_sage_linear_get_set_parameters() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    long num_nodes = 2048;
    long num_in_features = 1024;
    long num_out_features = 512;
    long num_params = 4;

    CudaHelper cuda_helper;
    SageLinear sage_linear(&cuda_helper, num_in_features, num_out_features, num_nodes);

    Matrix<float> **get_parameters_before = sage_linear.get_parameters();
    sage_linear.set_parameters(get_parameters_before);
    Matrix<float> **get_parameters_after = sage_linear.get_parameters();

    compare_mat(get_parameters_before[0], get_parameters_after[0], "self weights");
    compare_mat(get_parameters_before[1], get_parameters_after[1], "self bias");
    compare_mat(get_parameters_before[2], get_parameters_after[2], "neigh weights");
    compare_mat(get_parameters_before[3], get_parameters_after[3], "neigh bias");

    return 1;// TODO
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

    CHECK(test_sage_linear(&input_self, &input_neigh, &in_gradients, 1 << 15));
    CHECK(test_sage_linear(&input_self, &input_neigh, &in_gradients, 1 << 12));
    CHECK(test_sage_linear(&input_self, &input_neigh, &in_gradients, 1 << 8));
}

TEST_CASE("SageLinear, compare parameters", "[sagelinear][chunked][compare][parameters]") {
    CHECK(test_parameters());
}

TEST_CASE("SageLinear, compare", "[sagelinear][chunked][compare]") {
    CHECK(compare_sage_linear(1024));
    CHECK(compare_sage_linear(512));
    CHECK(compare_sage_linear(128));
}

TEST_CASE("SageLinear, set parameters", "[sagelinear][setparameters]") {
    CHECK(test_sage_linear_set_parameters());
}

TEST_CASE("SageLinear, get and set parameters", "[sagelinear][getsetparameters]") {
    CHECK(test_sage_linear_set_parameters());
}
