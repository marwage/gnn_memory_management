// Copyright 2020 Marcel Wagenländer

#include "adam.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "helper.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>


const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";


int test_sage_linear_adam(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *in_gradients) {
    std::string path;
    float learning_rate = 0.003;

    CudaHelper cuda_helper;
    SageLinear sage_linear_layer(&cuda_helper, input_self->num_columns_, in_gradients->num_columns_, input_self->num_rows_);
    std::vector<Matrix<float> *> params = sage_linear_layer.get_parameters();
    Adam adam(&cuda_helper, learning_rate, params, sage_linear_layer.get_gradients());

    save_params(params);

    Matrix<float> *activations = sage_linear_layer.forward(input_self, input_neigh);

    save_npy_matrix(activations, test_dir_path + "/activations.npy");

    SageLinearGradients *gradients = sage_linear_layer.backward(in_gradients);

    save_grads(gradients, sage_linear_layer.get_gradients());

    adam.step();

    save_npy_matrix(params[0], test_dir_path + "/self_weight_updated.npy");
    save_npy_matrix(params[1], test_dir_path + "/self_bias_updated.npy");
    save_npy_matrix(params[2], test_dir_path + "/neigh_weight_updated.npy");
    save_npy_matrix(params[3], test_dir_path + "/neigh_bias_updated.npy");

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear_adam.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_sage_linear_adam_chunked(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *incoming_gradients, long chunk_size) {
    std::string path;
    float learning_rate = 0.003;
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
    std::vector<Matrix<float> *> params = sage_linear_layer.get_parameters();
    Adam adam(&cuda_helper, learning_rate, params, sage_linear_layer.get_gradients());

    save_params(params);

    std::vector<Matrix<float>> *activations = sage_linear_layer.forward(&input_self_chunked, &input_neigh_chunked);

    Matrix<float> activations_one(num_nodes, num_out_features, false);
    stitch(activations, &activations_one);
    save_npy_matrix(&activations_one, test_dir_path + "/activations.npy");

    SageLinearGradientsChunked *gradients = sage_linear_layer.backward(&incoming_gradients_chunked);

    save_grads(gradients, sage_linear_layer.get_gradients(), num_nodes);

    adam.step();

    save_npy_matrix(params[0], test_dir_path + "/self_weight_updated.npy");
    save_npy_matrix(params[1], test_dir_path + "/self_bias_updated.npy");
    save_npy_matrix(params[2], test_dir_path + "/neigh_weight_updated.npy");
    save_npy_matrix(params[3], test_dir_path + "/neigh_bias_updated.npy");

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear_adam.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_sage_linear_adam_nans(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *in_gradients) {
    std::string path;
    float learning_rate = 0.003;
    bool has_nans = true;

    CudaHelper cuda_helper;
    SageLinear sage_linear_layer(&cuda_helper, input_self->num_columns_, in_gradients->num_columns_, input_self->num_rows_);
    std::vector<Matrix<float> *> parameters = sage_linear_layer.get_parameters();
    std::vector<Matrix<float> *> parameters_gradients = sage_linear_layer.get_gradients();
    Adam adam(&cuda_helper, learning_rate, parameters, parameters_gradients);

    long num_iterations = 3;
    for (long i = 0; i < num_iterations; ++i) {
        Matrix<float> *activations = sage_linear_layer.forward(input_self, input_neigh);

        has_nans = has_nans && check_nans(activations, "Activations");

        SageLinearGradients *gradients = sage_linear_layer.backward(in_gradients);

        has_nans = has_nans && check_nans(gradients->self_gradients, "Input self gradients");
        has_nans = has_nans && check_nans(gradients->neighbourhood_gradients, "Input neigh gradients");
        for (int i = 0; i < parameters_gradients.size(); ++i) {
            has_nans = has_nans && check_nans(parameters_gradients[i], "Parameter gradient " + std::to_string(i));
        }

        adam.step();

        for (int i = 0; i < parameters_gradients.size(); ++i) {
            has_nans = has_nans && check_nans(parameters[i], "Parameter " + std::to_string(i));
        }
    }

    return !has_nans;
}

int test_sage_linear_adam_nans_chunked(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *incoming_gradients, long chunk_size) {
    std::string path;
    float learning_rate = 0.003;
    bool has_nans = true;
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
    std::vector<Matrix<float> *> parameters = sage_linear_layer.get_parameters();
    std::vector<Matrix<float> *> parameters_gradients = sage_linear_layer.get_gradients();
    Adam adam(&cuda_helper, learning_rate, parameters, parameters_gradients);

    long num_iterations = 3;
    for (long i = 0; i < num_iterations; ++i) {
        std::vector<Matrix<float>> *activations = sage_linear_layer.forward(&input_self_chunked, &input_neigh_chunked);

        has_nans = has_nans && check_nans(activations, "Activations");

        SageLinearGradientsChunked *gradients = sage_linear_layer.backward(&incoming_gradients_chunked);

        has_nans = has_nans && check_nans(gradients->self_gradients, "Input self gradients");
        has_nans = has_nans && check_nans(gradients->neighbourhood_gradients, "Input neigh gradients");
        for (int i = 0; i < parameters_gradients.size(); ++i) {
            has_nans = has_nans && check_nans(parameters_gradients[i], "Parameter gradient " + std::to_string(i));
        }

        adam.step();

        for (int i = 0; i < parameters_gradients.size(); ++i) {
            has_nans = has_nans && check_nans(parameters[i], "Parameter " + std::to_string(i));
        }
    }

    return !has_nans;
}


TEST_CASE("SageLinear and Adam", "[sagelinear][adam]") {
    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    Matrix<float> input_self(rows, columns, true);
    input_self.set_random_values();
    save_npy_matrix(&input_self, test_dir_path + "/input_self.npy");

    Matrix<float> input_neigh(rows, columns, true);
    input_neigh.set_random_values();
    save_npy_matrix(&input_neigh, test_dir_path + "/input_neigh.npy");

    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_random_values();
    save_npy_matrix(&in_gradients, test_dir_path + "/in_gradients.npy");

    CHECK(test_sage_linear_adam(&input_self, &input_neigh, &in_gradients));
}

TEST_CASE("SageLinear and Adam chunked", "[sagelinear][adam][chunked]") {
    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    Matrix<float> input_self(rows, columns, true);
    input_self.set_random_values();
    save_npy_matrix(&input_self, test_dir_path + "/input_self.npy");

    Matrix<float> input_neigh(rows, columns, true);
    input_neigh.set_random_values();
    save_npy_matrix(&input_neigh, test_dir_path + "/input_neigh.npy");

    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_random_values();
    save_npy_matrix(&in_gradients, test_dir_path + "/in_gradients.npy");

    CHECK(test_sage_linear_adam_chunked(&input_self, &input_neigh, &in_gradients, 1 << 15));
    CHECK(test_sage_linear_adam_chunked(&input_self, &input_neigh, &in_gradients, 1 << 12));
    CHECK(test_sage_linear_adam_chunked(&input_self, &input_neigh, &in_gradients, 1 << 8));
}

TEST_CASE("SageLinear and Adam, NaNs", "[sagelinear][adam][nan]") {
    int rows = 2449029;// products
    int columns = 100; // products
    int num_out_features = 1 << 8;

    Matrix<float> input_self(rows, columns, true);
    input_self.set_random_values();

    Matrix<float> input_neigh(rows, columns, true);
    input_neigh.set_random_values();

    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_random_values();

    CHECK(test_sage_linear_adam_nans(&input_self, &input_neigh, &in_gradients));
}

TEST_CASE("SageLinear and Adam, chunked, NaNs", "[sagelinear][adam][chunked][nan]") {
    int rows = 2449029;// products
    int columns = 100; // products
    int num_out_features = 1 << 8;

    Matrix<float> input_self(rows, columns, true);
    input_self.set_random_values();

    Matrix<float> input_neigh(rows, columns, true);
    input_neigh.set_random_values();

    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_random_values();

    CHECK(test_sage_linear_adam_nans_chunked(&input_self, &input_neigh, &in_gradients, 1 << 15));
    CHECK(test_sage_linear_adam_nans_chunked(&input_self, &input_neigh, &in_gradients, 1 << 12));
    CHECK(test_sage_linear_adam_nans_chunked(&input_self, &input_neigh, &in_gradients, 1 << 8));
}