// Copyright 2020 Marcel Wagenländer

#include "log_softmax.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "dataset.hpp"
#include "gpu_memory_logger.hpp"
#include "tensors.hpp"

#include <benchmark/benchmark.h>

void benchmark_layer(Layer *layer, Dataset dataset, benchmark::State &state, bool forward);
void benchmark_layer_chunked(LayerChunked *layer, Dataset dataset, benchmark::State &state, bool forward);


static void BM_Layer_Logsoftmax_Flickr_Forward(benchmark::State &state) {
    LogSoftmax logsoftmax;
    benchmark_layer(&logsoftmax, flickr, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Flickr_Forward);

static void BM_Layer_Logsoftmax_Flickr_Backward(benchmark::State &state) {
    LogSoftmax logsoftmax;
    benchmark_layer(&logsoftmax, flickr, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Flickr_Backward);

static void BM_Layer_Logsoftmax_Reddit_Forward(benchmark::State &state) {
    LogSoftmax logsoftmax;
    benchmark_layer(&logsoftmax, reddit, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Reddit_Forward);

static void BM_Layer_Logsoftmax_Reddit_Backward(benchmark::State &state) {
    LogSoftmax logsoftmax;
    benchmark_layer(&logsoftmax, reddit, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Reddit_Backward);

static void BM_Layer_Logsoftmax_Products_Forward(benchmark::State &state) {
    LogSoftmax logsoftmax;
    benchmark_layer(&logsoftmax, products, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Products_Forward);

static void BM_Layer_Logsoftmax_Products_Backward(benchmark::State &state) {
    LogSoftmax logsoftmax;
    benchmark_layer(&logsoftmax, products, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Products_Backward);

// CHUNKED --- CHUNKED --- CHUNKED

static void BM_Layer_Logsoftmax_Chunked_Flickr_Forward(benchmark::State &state) {
    LogSoftmaxChunked logsoftmax;
    benchmark_layer_chunked(&logsoftmax, flickr, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Chunked_Flickr_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 15);

static void BM_Layer_Logsoftmax_Flickr_Chunked_Backward(benchmark::State &state) {
    LogSoftmaxChunked logsoftmax;
    benchmark_layer_chunked(&logsoftmax, flickr, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Flickr_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 15);

static void BM_Layer_Logsoftmax_Reddit_Chunked_Forward(benchmark::State &state) {
    LogSoftmaxChunked logsoftmax;
    benchmark_layer_chunked(&logsoftmax, reddit, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Reddit_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_Layer_Logsoftmax_Reddit_Chunked_Backward(benchmark::State &state) {
    LogSoftmaxChunked logsoftmax;
    benchmark_layer_chunked(&logsoftmax, reddit, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Reddit_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_Layer_Logsoftmax_Products_Chunked_Forward(benchmark::State &state) {
    LogSoftmaxChunked logsoftmax;
    benchmark_layer_chunked(&logsoftmax, products, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Products_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 21);

static void BM_Layer_Logsoftmax_Products_Chunked_Backward(benchmark::State &state) {
    LogSoftmaxChunked logsoftmax;
    benchmark_layer_chunked(&logsoftmax, products, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Products_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 21);

static void BM_Layer_Logsoftmax_Ivy_Chunked_Forward(benchmark::State &state) {
    LogSoftmaxChunked logsoftmax;
    benchmark_layer_chunked(&logsoftmax, ivy, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Ivy_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);

static void BM_Layer_Logsoftmax_Ivy_Chunked_Backward(benchmark::State &state) {
    LogSoftmaxChunked logsoftmax;
    benchmark_layer_chunked(&logsoftmax, ivy, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Ivy_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);

// PIPELINED --- PIPELINED --- PIPELINED

static void BM_Layer_Logsoftmax_Pipelined_Flickr_Forward(benchmark::State &state) {
    LogSoftmaxPipelined logsoftmax;
    benchmark_layer_chunked(&logsoftmax, flickr, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Pipelined_Flickr_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 15);

static void BM_Layer_Logsoftmax_Flickr_Pipelined_Backward(benchmark::State &state) {
    LogSoftmaxPipelined logsoftmax;
    benchmark_layer_chunked(&logsoftmax, flickr, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Flickr_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 15);

static void BM_Layer_Logsoftmax_Reddit_Pipelined_Forward(benchmark::State &state) {
    LogSoftmaxPipelined logsoftmax;
    benchmark_layer_chunked(&logsoftmax, reddit, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Reddit_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_Layer_Logsoftmax_Reddit_Pipelined_Backward(benchmark::State &state) {
    LogSoftmaxPipelined logsoftmax;
    benchmark_layer_chunked(&logsoftmax, reddit, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Reddit_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_Layer_Logsoftmax_Products_Pipelined_Forward(benchmark::State &state) {
    LogSoftmaxPipelined logsoftmax;
    benchmark_layer_chunked(&logsoftmax, products, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Products_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 21);

static void BM_Layer_Logsoftmax_Products_Pipelined_Backward(benchmark::State &state) {
    LogSoftmaxPipelined logsoftmax;
    benchmark_layer_chunked(&logsoftmax, products, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Products_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 21);

static void BM_Layer_Logsoftmax_Ivy_Pipelined_Forward(benchmark::State &state) {
    LogSoftmaxPipelined logsoftmax;
    benchmark_layer_chunked(&logsoftmax, ivy, state, true);
}
BENCHMARK(BM_Layer_Logsoftmax_Ivy_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);

static void BM_Layer_Logsoftmax_Ivy_Pipelined_Backward(benchmark::State &state) {
    LogSoftmaxPipelined logsoftmax;
    benchmark_layer_chunked(&logsoftmax, ivy, state, false);
}
BENCHMARK(BM_Layer_Logsoftmax_Ivy_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);
