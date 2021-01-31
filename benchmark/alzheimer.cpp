// Copyright 2020 Marcel Wagenländer

#include "alzheimer.hpp"
#include "dataset.hpp"
#include "gpu_memory_logger.hpp"

#include <benchmark/benchmark.h>
#include <string>


void benchmark_alzheimer(Dataset dataset, benchmark::State &state) {
    GPUMemoryLogger memory_logger("alzheimer_" + get_dataset_name(dataset));

    memory_logger.start();

    for (auto _ : state)
        alzheimer(dataset);

    memory_logger.stop();
}

void benchmark_alzheimer_chunked(Dataset dataset, bool keep_allocation, benchmark::State &state) {
    GPUMemoryLogger memory_logger("alzheimer_chunked_" + get_dataset_name(dataset) + "_" + std::to_string(state.range(0)));
    memory_logger.start();

    for (auto _ : state)
        alzheimer_chunked(dataset, state.range(0), keep_allocation);

    memory_logger.stop();
}

void benchmark_alzheimer_pipelined(Dataset dataset, benchmark::State &state) {
    GPUMemoryLogger memory_logger("alzheimer_pipelined_" + get_dataset_name(dataset) + "_" + std::to_string(state.range(0)));
    memory_logger.start();

    for (auto _ : state)
        alzheimer_pipelined(dataset, state.range(0));

    memory_logger.stop();
}

// LAYER --- LAYER --- LAYER

static void BM_Alzheimer_Layer_Flickr(benchmark::State &state) {
    benchmark_alzheimer(flickr, state);
}
BENCHMARK(BM_Alzheimer_Layer_Flickr);

static void BM_Alzheimer_Layer_Reddit(benchmark::State &state) {
    benchmark_alzheimer(reddit, state);
}
BENCHMARK(BM_Alzheimer_Layer_Reddit);

static void BM_Alzheimer_Layer_Products(benchmark::State &state) {
    benchmark_alzheimer(products, state);
}
BENCHMARK(BM_Alzheimer_Layer_Products);

// CHUNKED --- CHUNKED --- CHUNKED

static void BM_Alzheimer_Chunked_Flickr(benchmark::State &state) {
    benchmark_alzheimer_chunked(flickr, false, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Flickr)->RangeMultiplier(2)->Range(1 << 14, 1 << 16);

static void BM_Alzheimer_Chunked_Reddit(benchmark::State &state) {
    benchmark_alzheimer_chunked(reddit, false, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Reddit)->RangeMultiplier(2)->Range(1 << 14, 1 << 17);

static void BM_Alzheimer_Chunked_Products(benchmark::State &state) {
    benchmark_alzheimer_chunked(products, false, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Products)->RangeMultiplier(2)->Range(1 << 14, 1 << 21);

static void BM_Alzheimer_Chunked_Ivy(benchmark::State &state) {
    benchmark_alzheimer_chunked(ivy, false, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Ivy)->RangeMultiplier(2)->Range(1 << 14, 1 << 20);

// CHUNKED & KEEP --- CHUNKED & KEEP --- CHUNKED & KEEP

static void BM_Alzheimer_Chunked_Keep_Flickr(benchmark::State &state) {
    benchmark_alzheimer_chunked(flickr, true, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Keep_Flickr)->RangeMultiplier(2)->Range(1 << 14, 1 << 16);

static void BM_Alzheimer_Chunked_Keep_Reddit(benchmark::State &state) {
    benchmark_alzheimer_chunked(reddit, true, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Keep_Reddit)->RangeMultiplier(2)->Range(1 << 14, 1 << 17);

static void BM_Alzheimer_Chunked_Keep_Products(benchmark::State &state) {
    benchmark_alzheimer_chunked(products, true, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Keep_Products)->RangeMultiplier(2)->Range(1 << 14, 1 << 21);

static void BM_Alzheimer_Chunked_Keep_Ivy(benchmark::State &state) {
    benchmark_alzheimer_chunked(ivy, true, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Keep_Ivy)->RangeMultiplier(2)->Range(1 << 14, 1 << 20);

// some chunk size

//static void BM_Alzheimer_Chunked_Flickr_X(benchmark::State &state) {
//    benchmark_alzheimer_chunked(flickr, true, state);
//}
//BENCHMARK(BM_Alzheimer_Chunked_Flickr_X)->Arg(655981);

static void BM_Alzheimer_Chunked_Reddit_X(benchmark::State &state) {
    benchmark_alzheimer_chunked(reddit, true, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Reddit_X)->Arg(232965);

static void BM_Alzheimer_Chunked_Products_X(benchmark::State &state) {
    benchmark_alzheimer_chunked(products, true, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Products_X)->Arg(344896);

static void BM_Alzheimer_Chunked_Ivy_X(benchmark::State &state) {
    benchmark_alzheimer_chunked(ivy, true, state);
}
BENCHMARK(BM_Alzheimer_Chunked_Ivy_X)->Arg(330989);

// PIPELINED --- PIPELINED --- PIPELINED

static void BM_Alzheimer_Pipelined_Flickr(benchmark::State &state) {
    benchmark_alzheimer_pipelined(flickr, state);
}
BENCHMARK(BM_Alzheimer_Pipelined_Flickr)->RangeMultiplier(2)->Range(1 << 14, 1 << 16);

static void BM_Alzheimer_Pipelined_Reddit(benchmark::State &state) {
    benchmark_alzheimer_pipelined(reddit, state);
}
BENCHMARK(BM_Alzheimer_Pipelined_Reddit)->RangeMultiplier(2)->Range(1 << 14, 1 << 17);

static void BM_Alzheimer_Pipelined_Products(benchmark::State &state) {
    benchmark_alzheimer_pipelined(products, state);
}
BENCHMARK(BM_Alzheimer_Pipelined_Products)->RangeMultiplier(2)->Range(1 << 14, 1 << 20);

static void BM_Alzheimer_Pipelined_Ivy(benchmark::State &state) {
    benchmark_alzheimer_pipelined(ivy, state);
}
BENCHMARK(BM_Alzheimer_Pipelined_Ivy)->RangeMultiplier(2)->Range(1 << 14, 1 << 19);
