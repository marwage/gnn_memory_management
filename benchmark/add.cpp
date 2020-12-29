// Copyright 2020 Marcel Wagenländer

#include "add.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "dataset.hpp"
#include "gpu_memory_logger.hpp"
#include "tensors.hpp"

#include <benchmark/benchmark.h>

const std::string dir_path = "/mnt/data";


void benchmark_add(Dataset dataset, benchmark::State &state, bool forward) {
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);
    std::string path = dataset_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    Matrix<float> features_b(features.num_rows_, features.num_columns_, false);
    features_b.set_random_values();
    Matrix<float> incoming_gradients;
    if (!forward) {
        incoming_gradients.set(features.num_rows_, features.num_columns_, true);
        incoming_gradients.set_random_values();
    }

    CudaHelper cuda_helper;
    Add add(&cuda_helper, features.num_columns_, features.num_rows_);

    if (!forward) {
        add.forward(&features, &features_b);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(add.name_ + "_" + get_dataset_name(dataset) + "_" + direction);
    memory_logger.start();

    for (auto _ : state) {
        if (forward) {
            add.forward(&features, &features_b);
        } else {
            add.backward(&incoming_gradients);
        }
    }

    memory_logger.stop();
}

void benchmark_add_chunked(AddChunked *add, Dataset dataset, benchmark::State &state, bool forward) {
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);
    std::string path = dataset_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    Matrix<float> features_b(features.num_rows_, features.num_columns_, false);
    features_b.set_random_values();
    Matrix<float> incoming_gradients;
    if (!forward) {
        incoming_gradients.set(features.num_rows_, features.num_columns_, true);
        incoming_gradients.set_random_values();
    }

    long chunk_size = state.range(0);
    long num_chunks = ceil((float) features.num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    std::vector<Matrix<float>> features_b_chunked(num_chunks);
    chunk_up(&features_b, &features_b_chunked, chunk_size);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    if (!forward) {
        chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);
    }

    CudaHelper cuda_helper;
    add->set(&cuda_helper, chunk_size, features.num_rows_, features.num_columns_);

    if (!forward) {
        add->forward(&features_chunked, &features_b_chunked);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(add->name_ + "_" + get_dataset_name(dataset) + "_" + direction + "_" + std::to_string(chunk_size));
    memory_logger.start();

    for (auto _ : state) {
        if (forward) {
            add->forward(&features_chunked, &features_b_chunked);
        } else {
            add->backward(&incoming_gradients_chunked);
        }
    }

    memory_logger.stop();
}

static void BM_Layer_Add_Flickr_Forward(benchmark::State &state) {
    benchmark_add(flickr, state, true);
}
BENCHMARK(BM_Layer_Add_Flickr_Forward);

static void BM_Layer_Add_Reddit_Forward(benchmark::State &state) {
    benchmark_add(reddit, state, true);
}
BENCHMARK(BM_Layer_Add_Reddit_Forward);


static void BM_Layer_Add_Products_Forward(benchmark::State &state) {
    benchmark_add(products, state, true);
}
BENCHMARK(BM_Layer_Add_Products_Forward);

// CHUNKED --- CHUNKED --- CHUNKED

static void BM_Layer_Add_Flickr_Chunked_Forward(benchmark::State &state) {
    AddChunked add;
    benchmark_add_chunked(&add, flickr, state, true);
}
BENCHMARK(BM_Layer_Add_Flickr_Chunked_Forward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Add_Reddit_Chunked_Forward(benchmark::State &state) {
    AddChunked add;
    benchmark_add_chunked(&add, reddit, state, true);
}
BENCHMARK(BM_Layer_Add_Reddit_Chunked_Forward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Add_Products_Chunked_Forward(benchmark::State &state) {
    AddChunked add;
    benchmark_add_chunked(&add, products, state, true);
}
BENCHMARK(BM_Layer_Add_Products_Chunked_Forward)->Range(1 << 16, 1 << 21);

static void BM_Layer_Add_Ivy_Chunked_Forward(benchmark::State &state) {
    AddChunked add;
    benchmark_add_chunked(&add, ivy, state, true);
}
BENCHMARK(BM_Layer_Add_Ivy_Chunked_Forward)->Arg(1 << 19);

// PIPELINED --- PIPELINED --- PIPELINED

static void BM_Layer_Add_Flickr_Pipelined_Forward(benchmark::State &state) {
    AddPipelined add;
    benchmark_add_chunked(&add, flickr, state, true);
}
BENCHMARK(BM_Layer_Add_Flickr_Pipelined_Forward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Add_Reddit_Pipelined_Forward(benchmark::State &state) {
    AddPipelined add;
    benchmark_add_chunked(&add, reddit, state, true);
}
BENCHMARK(BM_Layer_Add_Reddit_Pipelined_Forward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Add_Products_Pipelined_Forward(benchmark::State &state) {
    AddPipelined add;
    benchmark_add_chunked(&add, products, state, true);
}
BENCHMARK(BM_Layer_Add_Products_Pipelined_Forward)->Range(1 << 16, 1 << 21);

static void BM_Layer_Add_Ivy_Pipelined_Forward(benchmark::State &state) {
    AddPipelined add;
    benchmark_add_chunked(&add, ivy, state, true);
}
BENCHMARK(BM_Layer_Add_Ivy_Pipelined_Forward)->Arg(1 << 19);
