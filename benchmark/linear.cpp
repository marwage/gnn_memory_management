// Copyright 2020 Marcel Wagenländer

#include "linear.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "dataset.hpp"
#include "gpu_memory_logger.hpp"
#include "tensors.hpp"

#include <benchmark/benchmark.h>

const std::string dir_path = "/mnt/data";
const long num_out_features = 256;


void benchmark_linear(Dataset dataset, benchmark::State &state, bool forward) {
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);
    std::string path = dataset_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    Matrix<float> incoming_gradients;
    if (!forward) {
        incoming_gradients.set(features.num_rows_, num_out_features, true);
        incoming_gradients.set_random_values();
    }

    CudaHelper cuda_helper;
    Linear linear(&cuda_helper, features.num_columns_, num_out_features, features.num_rows_);

    if (!forward) {
        linear.forward(&features);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(linear.name_ + "_" + get_dataset_name(dataset) + "_" + direction);
    memory_logger.start();

    for (auto _ : state) {
        if (forward) {
            linear.forward(&features);
        } else {
            linear.backward(&incoming_gradients);
        }
    }

    memory_logger.stop();
}

void benchmark_linear_chunked(LinearChunked *linear, Dataset dataset, benchmark::State &state, bool forward) {
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);
    std::string path = dataset_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    Matrix<float> incoming_gradients;
    if (!forward) {
        incoming_gradients.set(features.num_rows_, num_out_features, true);
        incoming_gradients.set_random_values();
    }

    long chunk_size = state.range(0);
    long num_chunks = ceil((float) features.num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    if (!forward) {
        chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);
    }

    CudaHelper cuda_helper;
    linear->set(&cuda_helper, chunk_size, features.num_rows_, features.num_columns_, num_out_features);

    if (!forward) {
        linear->forward(&features_chunked);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(linear->name_ + "_" + get_dataset_name(dataset) + "_" + direction + "_" + std::to_string(chunk_size));
    memory_logger.start();

    for (auto _ : state) {
        if (forward) {
            linear->forward(&features_chunked);
        } else {
            linear->backward(&incoming_gradients_chunked);
        }
    }

    memory_logger.stop();
}

static void BM_Layer_Linear_Flickr_Forward(benchmark::State &state) {
    benchmark_linear(flickr, state, true);
}
BENCHMARK(BM_Layer_Linear_Flickr_Forward);

static void BM_Layer_Linear_Flickr_Backward(benchmark::State &state) {
    benchmark_linear(flickr, state, false);
}
BENCHMARK(BM_Layer_Linear_Flickr_Backward);

static void BM_Layer_Linear_Reddit_Forward(benchmark::State &state) {
    benchmark_linear(reddit, state, true);
}
BENCHMARK(BM_Layer_Linear_Reddit_Forward);

static void BM_Layer_Linear_Reddit_Backward(benchmark::State &state) {
    benchmark_linear(reddit, state, false);
}
BENCHMARK(BM_Layer_Linear_Reddit_Backward);

static void BM_Layer_Linear_Products_Forward(benchmark::State &state) {
    benchmark_linear(products, state, true);
}
BENCHMARK(BM_Layer_Linear_Products_Forward);

static void BM_Layer_Linear_Products_Backward(benchmark::State &state) {
    benchmark_linear(products, state, false);
}
BENCHMARK(BM_Layer_Linear_Products_Backward);

// CHUNKED --- CHUNKED --- CHUNKED

static void BM_Layer_Linear_Flickr_Chunked_Forward(benchmark::State &state) {
    LinearChunked linear;
    benchmark_linear_chunked(&linear, flickr, state, true);
}
BENCHMARK(BM_Layer_Linear_Flickr_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 12, 1 << 16);

static void BM_Layer_Linear_Flickr_Chunked_Backward(benchmark::State &state) {
    LinearChunked linear;
    benchmark_linear_chunked(&linear, flickr, state, false);
}
BENCHMARK(BM_Layer_Linear_Flickr_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 12, 1 << 16);

static void BM_Layer_Linear_Reddit_Chunked_Forward(benchmark::State &state) {
    LinearChunked linear;
    benchmark_linear_chunked(&linear, reddit, state, true);
}
BENCHMARK(BM_Layer_Linear_Reddit_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 12, 1 << 17);

static void BM_Layer_Linear_Reddit_Chunked_Backward(benchmark::State &state) {
    LinearChunked linear;
    benchmark_linear_chunked(&linear, reddit, state, false);
}
BENCHMARK(BM_Layer_Linear_Reddit_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 12, 1 << 17);

static void BM_Layer_Linear_Products_Chunked_Forward(benchmark::State &state) {
    LinearChunked linear;
    benchmark_linear_chunked(&linear, products, state, true);
}
BENCHMARK(BM_Layer_Linear_Products_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 12, 1 << 21);

static void BM_Layer_Linear_Products_Chunked_Backward(benchmark::State &state) {
    LinearChunked linear;
    benchmark_linear_chunked(&linear, products, state, false);
}
BENCHMARK(BM_Layer_Linear_Products_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 12, 1 << 21);

static void BM_Layer_Linear_Ivy_Chunked_Forward(benchmark::State &state) {
    LinearChunked linear;
    benchmark_linear_chunked(&linear, ivy, state, true);
}
BENCHMARK(BM_Layer_Linear_Ivy_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 12, 1 << 19);

static void BM_Layer_Linear_Ivy_Chunked_Backward(benchmark::State &state) {
    LinearChunked linear;
    benchmark_linear_chunked(&linear, ivy, state, false);
}
BENCHMARK(BM_Layer_Linear_Ivy_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 12, 1 << 19);

// PIPELINED --- PIPELINED --- PIPELINED

static void BM_Layer_Linear_Flickr_Pipelined_Forward(benchmark::State &state) {
    LinearPipelined linear;
    benchmark_linear_chunked(&linear, flickr, state, true);
}
BENCHMARK(BM_Layer_Linear_Flickr_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 12, 1 << 16);

static void BM_Layer_Linear_Flickr_Pipelined_Backward(benchmark::State &state) {
    LinearPipelined linear;
    benchmark_linear_chunked(&linear, flickr, state, false);
}
BENCHMARK(BM_Layer_Linear_Flickr_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 12, 1 << 16);

static void BM_Layer_Linear_Reddit_Pipelined_Forward(benchmark::State &state) {
    LinearPipelined linear;
    benchmark_linear_chunked(&linear, reddit, state, true);
}
BENCHMARK(BM_Layer_Linear_Reddit_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 12, 1 << 17);

static void BM_Layer_Linear_Reddit_Pipelined_Backward(benchmark::State &state) {
    LinearPipelined linear;
    benchmark_linear_chunked(&linear, reddit, state, false);
}
BENCHMARK(BM_Layer_Linear_Reddit_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 12, 1 << 17);

static void BM_Layer_Linear_Products_Pipelined_Forward(benchmark::State &state) {
    LinearPipelined linear;
    benchmark_linear_chunked(&linear, products, state, true);
}
BENCHMARK(BM_Layer_Linear_Products_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 12, 1 << 21);

static void BM_Layer_Linear_Products_Pipelined_Backward(benchmark::State &state) {
    LinearPipelined linear;
    benchmark_linear_chunked(&linear, products, state, false);
}
BENCHMARK(BM_Layer_Linear_Products_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 12, 1 << 21);

static void BM_Layer_Linear_Ivy_Pipelined_Forward(benchmark::State &state) {
    LinearPipelined linear;
    benchmark_linear_chunked(&linear, ivy, state, true);
}
BENCHMARK(BM_Layer_Linear_Ivy_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 12, 1 << 19);

static void BM_Layer_Linear_Ivy_Pipelined_Backward(benchmark::State &state) {
    LinearPipelined linear;
    benchmark_linear_chunked(&linear, ivy, state, false);
}
BENCHMARK(BM_Layer_Linear_Ivy_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 12, 1 << 19);
