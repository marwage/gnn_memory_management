// Copyright 2020 Marcel Wagenländer

#include "relu.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "gpu_memory_logger.hpp"
#include "tensors.hpp"
#include "dataset.hpp"

#include <benchmark/benchmark.h>

void benchmark_layer(Layer *layer, Dataset dataset, benchmark::State &state, bool forward);
void benchmark_layer_chunked(LayerChunked *layer, Dataset dataset, benchmark::State &state, bool forward);


static void BM_Layer_Relu_Flickr_Forward(benchmark::State &state) {
    Relu relu;
    benchmark_layer(&relu, flickr, state, true);
}
BENCHMARK(BM_Layer_Relu_Flickr_Forward);

static void BM_Layer_Relu_Flickr_Backward(benchmark::State &state) {
    Relu relu;
    benchmark_layer(&relu, flickr, state, false);
}
BENCHMARK(BM_Layer_Relu_Flickr_Backward);

static void BM_Layer_Relu_Reddit_Forward(benchmark::State &state) {
    Relu relu;
    benchmark_layer(&relu, reddit, state, true);
}
BENCHMARK(BM_Layer_Relu_Reddit_Forward);

static void BM_Layer_Relu_Reddit_Backward(benchmark::State &state) {
    Relu relu;
    benchmark_layer(&relu, reddit, state, false);
}
BENCHMARK(BM_Layer_Relu_Reddit_Backward);

static void BM_Layer_Relu_Products_Forward(benchmark::State &state) {
    Relu relu;
    benchmark_layer(&relu, products, state, true);
}
BENCHMARK(BM_Layer_Relu_Products_Forward);

static void BM_Layer_Relu_Products_Backward(benchmark::State &state) {
    Relu relu;
    benchmark_layer(&relu, products, state, false);
}
BENCHMARK(BM_Layer_Relu_Products_Backward);

static void BM_Layer_Relu_Products_Backward_Hidden(benchmark::State &state) {
    long num_nodes = 2449029;
    long num_features = 256;
    Matrix<float> input(num_nodes, num_features, true);
    input.set_random_values();
    Matrix<float> incoming_gradients(num_nodes, num_features, true);
    incoming_gradients.set_random_values();

    CudaHelper cuda_helper;
    Relu relu(&cuda_helper, num_nodes, num_features);

    Matrix<float> *activations = relu.forward(&input);

    GPUMemoryLogger memory_logger("relu_products_backward_hidden", 50);
    memory_logger.start();

    Matrix<float> *gradients;
    for (auto _ : state) {
        gradients = relu.backward(&incoming_gradients);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Products_Backward_Hidden);

static void BM_Layer_Relu_Products_Chunked_Backward_Hidden(benchmark::State &state) {
    long num_nodes = 2449029;
    long num_features = 256;
    Matrix<float> input(num_nodes, num_features, true);
    input.set_random_values();
    Matrix<float> incoming_gradients(num_nodes, num_features, true);
    incoming_gradients.set_random_values();

    long chunk_size = state.range(0);
    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    std::vector<Matrix<float>> input_chunked(num_chunks);
    chunk_up(&input, &input_chunked, chunk_size);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);

    CudaHelper cuda_helper;
    ReluChunked relu(&cuda_helper, chunk_size, num_nodes, num_features);

    std::vector<Matrix<float>> *activations = relu.forward(&input_chunked);

    GPUMemoryLogger memory_logger("relu_products_backward_hidden_" + std::to_string(chunk_size));
    memory_logger.start();

    std::vector<Matrix<float>> *gradients;
    for (auto _ : state) {
        gradients = relu.backward(&incoming_gradients_chunked);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Relu_Products_Chunked_Backward_Hidden)->Range(1 << 16, 1 << 21);

// CHUNKED --- CHUNKED --- CHUNKED

static void BM_Layer_Relu_Reddit_Chunked_Constructor(benchmark::State &state) {
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
BENCHMARK(BM_Layer_Relu_Reddit_Chunked_Constructor)->Range(1 << 12, 1 << 17);

static void BM_Layer_Relu_Chunked_Flickr_Forward(benchmark::State &state) {
    ReluChunked relu;
    benchmark_layer_chunked(&relu, flickr, state, true);
}
BENCHMARK(BM_Layer_Relu_Chunked_Flickr_Forward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Relu_Flickr_Chunked_Backward(benchmark::State &state) {
    ReluChunked relu;
    benchmark_layer_chunked(&relu, flickr, state, false);
}
BENCHMARK(BM_Layer_Relu_Flickr_Chunked_Backward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Relu_Reddit_Chunked_Forward(benchmark::State &state) {
    ReluChunked relu;
    benchmark_layer_chunked(&relu, reddit, state, true);
}
BENCHMARK(BM_Layer_Relu_Reddit_Chunked_Forward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Relu_Reddit_Chunked_Backward(benchmark::State &state) {
    ReluChunked relu;
    benchmark_layer_chunked(&relu, reddit, state, false);
}
BENCHMARK(BM_Layer_Relu_Reddit_Chunked_Backward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Relu_Products_Chunked_Forward(benchmark::State &state) {
    ReluChunked relu;
    benchmark_layer_chunked(&relu, products, state, true);
}
BENCHMARK(BM_Layer_Relu_Products_Chunked_Forward)->Range(1 << 16, 1 << 21);

static void BM_Layer_Relu_Products_Chunked_Backward(benchmark::State &state) {
    ReluChunked relu;
    benchmark_layer_chunked(&relu, products, state, false);
}
BENCHMARK(BM_Layer_Relu_Products_Chunked_Backward)->Range(1 << 16, 1 << 21);

// PIPELINED --- PIPELINED --- PIPELINED

static void BM_Layer_Relu_Pipelined_Flickr_Forward(benchmark::State &state) {
    ReluPipelined relu;
    benchmark_layer_chunked(&relu, flickr, state, true);
}
BENCHMARK(BM_Layer_Relu_Pipelined_Flickr_Forward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Relu_Flickr_Pipelined_Backward(benchmark::State &state) {
    ReluPipelined relu;
    benchmark_layer_chunked(&relu, flickr, state, false);
}
BENCHMARK(BM_Layer_Relu_Flickr_Pipelined_Backward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Relu_Reddit_Pipelined_Forward(benchmark::State &state) {
    ReluPipelined relu;
    benchmark_layer_chunked(&relu, reddit, state, true);
}
BENCHMARK(BM_Layer_Relu_Reddit_Pipelined_Forward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Relu_Reddit_Pipelined_Backward(benchmark::State &state) {
    ReluPipelined relu;
    benchmark_layer_chunked(&relu, reddit, state, false);
}
BENCHMARK(BM_Layer_Relu_Reddit_Pipelined_Backward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Relu_Products_Pipelined_Forward(benchmark::State &state) {
    ReluPipelined relu;
    benchmark_layer_chunked(&relu, products, state, true);
}
BENCHMARK(BM_Layer_Relu_Products_Pipelined_Forward)->Range(1 << 16, 1 << 21);

static void BM_Layer_Relu_Products_Pipelined_Backward(benchmark::State &state) {
    ReluPipelined relu;
    benchmark_layer_chunked(&relu, products, state, false);
}
BENCHMARK(BM_Layer_Relu_Products_Pipelined_Backward)->Range(1 << 16, 1 << 21);
