// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
#include "cuda_helper.hpp"
#include "gpu_memory_logger.hpp"
#include "tensors.hpp"

#include <benchmark/benchmark.h>


const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string reddit_dir_path = dir_path + "/reddit";
const std::string products_dir_path = dir_path + "/products";

const long num_out_features = 512;


static void BM_Layer_Sagelinear_Flickr_Forward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, false);
    aggregates.set_random_values();

    CudaHelper cuda_helper;
    SageLinear sagelinear(&cuda_helper, features.num_columns_, num_out_features, features.num_rows_);

    GPUMemoryLogger memory_logger("sagelinear_flickr_forward");
    memory_logger.start();

    Matrix<float> *activations;
    for (auto _ : state) {
        activations = sagelinear.forward(&features, &aggregates);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Flickr_Forward);

static void BM_Layer_Sagelinear_Chunked_Flickr_Forward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, true);
    aggregates.set_random_values();

    CudaHelper cuda_helper;
    SageLinearChunked sagelinear(&cuda_helper, features.num_columns_, num_out_features, state.range(0), features.num_rows_);

    GPUMemoryLogger memory_logger("sagelinear_flickr_forward_" + std::to_string(state.range(0)));
    memory_logger.start();

    Matrix<float> *activations;
    for (auto _ : state) {
        activations = sagelinear.forward(&features, &aggregates);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Chunked_Flickr_Forward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Sagelinear_Flickr_Backward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, false);
    aggregates.set_random_values();
    Matrix<float> in_gradients(features.num_rows_, num_out_features, false);
    in_gradients.set_random_values();

    CudaHelper cuda_helper;
    SageLinear sagelinear(&cuda_helper, features.num_columns_, num_out_features, features.num_rows_);

    Matrix<float> *activations = sagelinear.forward(&features, &aggregates);

    GPUMemoryLogger memory_logger("sagelinear_flickr_backward");
    memory_logger.start();

    SageLinearGradients *gradients;
    for (auto _ : state) {
        gradients = sagelinear.backward(&in_gradients);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Flickr_Backward);

static void BM_Layer_Sagelinear_Flickr_Chunked_Backward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, true);
    aggregates.set_random_values();
    Matrix<float> in_gradients(features.num_rows_, num_out_features, true);
    in_gradients.set_random_values();

    CudaHelper cuda_helper;
    SageLinearChunked sagelinear(&cuda_helper, features.num_columns_, num_out_features, state.range(0), features.num_rows_);

    Matrix<float> *activations = sagelinear.forward(&features, &aggregates);

    GPUMemoryLogger memory_logger("sagelinear_flickr_backward_" + std::to_string(state.range(0)));
    memory_logger.start();

    SageLinearGradients *gradients;
    for (auto _ : state) {
        gradients = sagelinear.backward(&in_gradients);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Flickr_Chunked_Backward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Sagelinear_Reddit_Forward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, false);
    aggregates.set_random_values();

    CudaHelper cuda_helper;
    SageLinear sagelinear(&cuda_helper, features.num_columns_, num_out_features, features.num_rows_);

    GPUMemoryLogger memory_logger("sagelinear_reddit_forward");
    memory_logger.start();

    Matrix<float> *activations;
    for (auto _ : state) {
        activations = sagelinear.forward(&features, &aggregates);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Reddit_Forward);

static void BM_Layer_Sagelinear_Reddit_Chunked_Forward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, true);
    aggregates.set_random_values();

    CudaHelper cuda_helper;
    SageLinearChunked sagelinear(&cuda_helper, features.num_columns_, num_out_features, state.range(0), features.num_rows_);

    GPUMemoryLogger memory_logger("sagelinear_reddit_forward_" + std::to_string(state.range(0)));
    memory_logger.start();

    Matrix<float> *activations;
    for (auto _ : state) {
        activations = sagelinear.forward(&features, &aggregates);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Reddit_Chunked_Forward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Sagelinar_Reddit_Backward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, false);
    aggregates.set_random_values();
    Matrix<float> in_gradients(features.num_rows_, num_out_features, false);
    in_gradients.set_random_values();

    CudaHelper cuda_helper;
    SageLinear sagelinear(&cuda_helper, features.num_columns_, num_out_features, features.num_rows_);

    Matrix<float> *activations = sagelinear.forward(&features, &aggregates);

    GPUMemoryLogger memory_logger("sagelinear_reddit_backward");
    memory_logger.start();

    SageLinearGradients *gradients;
    for (auto _ : state) {
        gradients = sagelinear.backward(&in_gradients);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinar_Reddit_Backward);

static void BM_Layer_Sagelinear_Reddit_Chunked_Backward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, true);
    aggregates.set_random_values();
    Matrix<float> in_gradients(features.num_rows_, num_out_features, true);
    in_gradients.set_random_values();

    CudaHelper cuda_helper;
    SageLinearChunked sagelinear(&cuda_helper, features.num_columns_, num_out_features, state.range(0), features.num_rows_);

    Matrix<float> *activations = sagelinear.forward(&features, &aggregates);

    GPUMemoryLogger memory_logger("sagelinear_reddit_backward_" + std::to_string(state.range(0)));
    memory_logger.start();

    SageLinearGradients *gradients;
    for (auto _ : state) {
        gradients = sagelinear.backward(&in_gradients);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Reddit_Chunked_Backward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Sagelinear_Products_Forward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, false);
    aggregates.set_random_values();

    CudaHelper cuda_helper;
    SageLinear sagelinear(&cuda_helper, features.num_columns_, num_out_features, features.num_rows_);

    GPUMemoryLogger memory_logger("sagelinear_products_forward");
    memory_logger.start();

    Matrix<float> *activations;
    for (auto _ : state) {
        activations = sagelinear.forward(&features, &aggregates);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Products_Forward);

static void BM_Layer_Sagelinear_Products_Chunked_Forward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, true);
    aggregates.set_random_values();

    CudaHelper cuda_helper;
    SageLinearChunked sagelinear(&cuda_helper, features.num_columns_, num_out_features, state.range(0), features.num_rows_);

    GPUMemoryLogger memory_logger("sagelinear_products_forward_" + std::to_string(state.range(0)));
    memory_logger.start();

    Matrix<float> *activations;
    for (auto _ : state) {
        activations = sagelinear.forward(&features, &aggregates);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Products_Chunked_Forward)->Range(1 << 16, 1 << 21);

static void BM_Layer_Sagelinear_Products_Backward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, false);
    aggregates.set_random_values();
    Matrix<float> in_gradients(features.num_rows_, num_out_features, false);
    in_gradients.set_random_values();

    CudaHelper cuda_helper;
    SageLinear sagelinear(&cuda_helper, features.num_columns_, num_out_features, features.num_rows_);

    Matrix<float> *activations = sagelinear.forward(&features, &aggregates);

    GPUMemoryLogger memory_logger("graphconv_products_backward");
    memory_logger.start();

    SageLinearGradients *gradients;
    for (auto _ : state) {
        gradients = sagelinear.backward(&in_gradients);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Products_Backward);

static void BM_Layer_Sagelinear_Products_Chunked_Backward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, true);
    aggregates.set_random_values();
    Matrix<float> in_gradients(features.num_rows_, num_out_features, true);
    in_gradients.set_random_values();

    CudaHelper cuda_helper;
    SageLinearChunked sagelinear(&cuda_helper, features.num_columns_, num_out_features, state.range(0), features.num_rows_);

    Matrix<float> *activations = sagelinear.forward(&features, &aggregates);

    GPUMemoryLogger memory_logger("sagelinear_products_backward_" + std::to_string(state.range(0)));
    memory_logger.start();

    SageLinearGradients *gradients;
    for (auto _ : state) {
        gradients = sagelinear.backward(&in_gradients);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Sagelinear_Products_Chunked_Backward)->Range(1 << 16, 1 << 21);
