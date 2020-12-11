// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "gpu_memory_logger.hpp"
#include "tensors.hpp"

#include <benchmark/benchmark.h>


const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string reddit_dir_path = dir_path + "/reddit";
const std::string products_dir_path = dir_path + "/products";


static void BM_Layer_Relu_Chunked_Reddit_Constructor(benchmark::State &state) {
    // reddit dataset
    long num_nodes = 232965;
    long num_features = 602;
    CudaHelper cuda_helper;
    long chunk_size = state.range(0);

    GPUMemoryLogger memory_logger("relu_reddit_constructor_" + std::to_string(chunk_size));
    memory_logger.start();

    for (auto _ : state) {
        ReluChunked relu(&cuda_helper, chunk_size, num_nodes, num_features);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Chunked_Reddit_Constructor)->Range(1 << 12, 1 << 17);

static void BM_Layer_Relu_Flickr_Forward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    CudaHelper cuda_helper;
    Relu relu(&cuda_helper, features.num_rows_, features.num_columns_);

    GPUMemoryLogger memory_logger("relu_flickr_forward");
    memory_logger.start();

    Matrix<float> *activations;
    for (auto _ : state) {
        activations = relu.forward(&features);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Flickr_Forward);

static void BM_Layer_Relu_Chunked_Flickr_Forward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    long chunk_size = state.range(0);
    long num_chunks = ceil((float) features.num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);

    CudaHelper cuda_helper;
    ReluChunked relu(&cuda_helper, chunk_size, features.num_rows_, features.num_columns_);

    GPUMemoryLogger memory_logger("relu_flickr_forward_" + std::to_string(chunk_size));
    memory_logger.start();

    std::vector<Matrix<float>> *activations;
    for (auto _ : state) {
        activations = relu.forward(&features_chunked);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Chunked_Flickr_Forward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Relu_Flickr_Backward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> in_gradients(features.num_rows_, features.num_columns_, true);
    in_gradients.set_random_values();

    CudaHelper cuda_helper;
    Relu relu(&cuda_helper, features.num_rows_, features.num_columns_);

    Matrix<float> *activations = relu.forward(&features);

    GPUMemoryLogger memory_logger("relu_flickr_backward");
    memory_logger.start();

    Matrix<float> *gradients;
    for (auto _ : state) {
        gradients = relu.backward(&in_gradients);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Flickr_Backward);

static void BM_Layer_Relu_Flickr_Chunked_Backward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> in_gradients(features.num_rows_, features.num_columns_, true);
    in_gradients.set_random_values();

    long chunk_size = state.range(0);
    long num_chunks = ceil((float) features.num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    std::vector<Matrix<float>> in_gradients_chunked(num_chunks);
    chunk_up(&in_gradients, &in_gradients_chunked, chunk_size);

    CudaHelper cuda_helper;
    ReluChunked relu(&cuda_helper, state.range(0), features.num_rows_, features.num_columns_);

    std::vector<Matrix<float>> *activations = relu.forward(&features_chunked);

    GPUMemoryLogger memory_logger("relu_flickr_backward_" + std::to_string(state.range(0)));
    memory_logger.start();

    std::vector<Matrix<float>> *gradients;
    for (auto _ : state) {
        gradients = relu.backward(&in_gradients_chunked);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Flickr_Chunked_Backward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Relu_Reddit_Forward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    CudaHelper cuda_helper;
    Relu relu(&cuda_helper, features.num_rows_, features.num_columns_);

    GPUMemoryLogger memory_logger("relu_reddit_forward");
    memory_logger.start();

    Matrix<float> *activations;
    for (auto _ : state) {
        activations = relu.forward(&features);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Reddit_Forward);

static void BM_Layer_Relu_Reddit_Chunked_Forward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    long chunk_size = state.range(0);
    long num_chunks = ceil((float) features.num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);

    CudaHelper cuda_helper;
    ReluChunked relu(&cuda_helper, state.range(0), features.num_rows_, features.num_columns_);

    GPUMemoryLogger memory_logger("relu_reddit_forward_" + std::to_string(state.range(0)));
    memory_logger.start();

    std::vector<Matrix<float>> *activations;
    for (auto _ : state) {
        activations = relu.forward(&features_chunked);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Reddit_Chunked_Forward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Relu_Reddit_Backward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> in_gradients(features.num_rows_, features.num_columns_, true);
    in_gradients.set_random_values();

    CudaHelper cuda_helper;
    Relu relu(&cuda_helper, features.num_rows_, features.num_columns_);

    Matrix<float> *activations = relu.forward(&features);

    GPUMemoryLogger memory_logger("relu_reddit_backward");
    memory_logger.start();

    Matrix<float> *gradients;
    for (auto _ : state) {
        gradients = relu.backward(&in_gradients);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Reddit_Backward);

static void BM_Layer_Relu_Reddit_Chunked_Backward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> in_gradients(features.num_rows_, features.num_columns_, true);
    in_gradients.set_random_values();

    long chunk_size = state.range(0);
    long num_chunks = ceil((float) features.num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    std::vector<Matrix<float>> in_gradients_chunked(num_chunks);
    chunk_up(&in_gradients, &in_gradients_chunked, chunk_size);

    CudaHelper cuda_helper;
    ReluChunked relu(&cuda_helper, state.range(0), features.num_rows_, features.num_columns_);

    std::vector<Matrix<float>> *activations = relu.forward(&features_chunked);

    GPUMemoryLogger memory_logger("relu_reddit_backward_" + std::to_string(state.range(0)));
    memory_logger.start();

    std::vector<Matrix<float>> *gradients;
    for (auto _ : state) {
        gradients = relu.backward(&in_gradients_chunked);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Reddit_Chunked_Backward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Relu_Products_Forward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    CudaHelper cuda_helper;
    Relu relu(&cuda_helper, features.num_rows_, features.num_columns_);

    GPUMemoryLogger memory_logger("relu_products_forward");
    memory_logger.start();

    Matrix<float> *activations;
    for (auto _ : state) {
        activations = relu.forward(&features);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Products_Forward);

static void BM_Layer_Relu_Products_Chunked_Forward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    long chunk_size = state.range(0);
    long num_chunks = ceil((float) features.num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);

    CudaHelper cuda_helper;
    ReluChunked relu(&cuda_helper, state.range(0), features.num_rows_, features.num_columns_);

    GPUMemoryLogger memory_logger("relu_products_forward_" + std::to_string(state.range(0)));
    memory_logger.start();

    std::vector<Matrix<float>> *activations;
    for (auto _ : state) {
        activations = relu.forward(&features_chunked);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Products_Chunked_Forward)->Range(1 << 16, 1 << 21);

static void BM_Layer_Relu_Products_Backward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> in_gradients(features.num_rows_, features.num_columns_, true);
    in_gradients.set_random_values();

    CudaHelper cuda_helper;
    Relu relu(&cuda_helper, features.num_rows_, features.num_columns_);

    Matrix<float> *activations = relu.forward(&features);

    GPUMemoryLogger memory_logger("relu_products_backward");
    memory_logger.start();

    Matrix<float> *gradients;
    for (auto _ : state) {
        gradients = relu.backward(&in_gradients);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Products_Backward);

static void BM_Layer_Relu_Products_Chunked_Backward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> in_gradients(features.num_rows_, features.num_columns_, true);
    in_gradients.set_random_values();

    long chunk_size = state.range(0);
    long num_chunks = ceil((float) features.num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    std::vector<Matrix<float>> in_gradients_chunked(num_chunks);
    chunk_up(&in_gradients, &in_gradients_chunked, chunk_size);

    CudaHelper cuda_helper;
    ReluChunked relu(&cuda_helper, state.range(0), features.num_rows_, features.num_columns_);

    std::vector<Matrix<float>> *activations = relu.forward(&features_chunked);

    GPUMemoryLogger memory_logger("relu_products_backward_" + std::to_string(state.range(0)));
    memory_logger.start();

    std::vector<Matrix<float>> *gradients;
    for (auto _ : state) {
        gradients = relu.backward(&in_gradients_chunked);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Products_Chunked_Backward)->Range(1 << 16, 1 << 21);
