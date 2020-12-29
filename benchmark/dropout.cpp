// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "dataset.hpp"

#include <benchmark/benchmark.h>

void benchmark_layer(Layer *layer, Dataset dataset, benchmark::State &state, bool forward);
void benchmark_layer_chunked(LayerChunked *layer, Dataset dataset, benchmark::State &state, bool forward);


static void BM_Layer_Dropout_Flickr_Forward(benchmark::State &state) {
    Dropout dropout;
    benchmark_layer(&dropout, flickr, state, true);
}
BENCHMARK(BM_Layer_Dropout_Flickr_Forward);

static void BM_Layer_Dropout_Flickr_Backward(benchmark::State &state) {
    Dropout dropout;
    benchmark_layer(&dropout, flickr, state, false);
}
BENCHMARK(BM_Layer_Dropout_Flickr_Backward);

static void BM_Layer_Dropout_Reddit_Forward(benchmark::State &state) {
    Dropout dropout;
    benchmark_layer(&dropout, reddit, state, true);
}
BENCHMARK(BM_Layer_Dropout_Reddit_Forward);

static void BM_Layer_Dropout_Reddit_Backward(benchmark::State &state) {
    Dropout dropout;
    benchmark_layer(&dropout, reddit, state, false);
}
BENCHMARK(BM_Layer_Dropout_Reddit_Backward);

static void BM_Layer_Dropout_Products_Forward(benchmark::State &state) {
    Dropout dropout;
    benchmark_layer(&dropout, products, state, true);
}
BENCHMARK(BM_Layer_Dropout_Products_Forward);

static void BM_Layer_Dropout_Products_Backward(benchmark::State &state) {
    Dropout dropout;
    benchmark_layer(&dropout, products, state, false);
}
BENCHMARK(BM_Layer_Dropout_Products_Backward);

// CHUNKED --- CHUNKED --- CHUNKED

static void BM_Layer_Dropout_Flickr_Chunked_Forward(benchmark::State &state) {
    DropoutChunked dropout;
    benchmark_layer_chunked(&dropout, flickr, state, true);
}
BENCHMARK(BM_Layer_Dropout_Flickr_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 15);

static void BM_Layer_Dropout_Flickr_Chunked_Backward(benchmark::State &state) {
    DropoutChunked dropout;
    benchmark_layer_chunked(&dropout, flickr, state, false);
}
BENCHMARK(BM_Layer_Dropout_Flickr_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 15);

static void BM_Layer_Dropout_Reddit_Chunked_Forward(benchmark::State &state) {
    DropoutChunked dropout;
    benchmark_layer_chunked(&dropout, reddit, state, true);
}
BENCHMARK(BM_Layer_Dropout_Reddit_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_Layer_Dropout_Reddit_Chunked_Backward(benchmark::State &state) {
    DropoutChunked dropout;
    benchmark_layer_chunked(&dropout, reddit, state, false);
}
BENCHMARK(BM_Layer_Dropout_Reddit_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_Layer_Dropout_Products_Chunked_Forward(benchmark::State &state) {
    DropoutChunked dropout;
    benchmark_layer_chunked(&dropout, products, state, true);
}
BENCHMARK(BM_Layer_Dropout_Products_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 21);

static void BM_Layer_Dropout_Products_Chunked_Backward(benchmark::State &state) {
    DropoutChunked dropout;
    benchmark_layer_chunked(&dropout, products, state, false);
}
BENCHMARK(BM_Layer_Dropout_Products_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 21);

static void BM_Layer_Dropout_Ivy_Chunked_Forward(benchmark::State &state) {
    DropoutChunked dropout;
    benchmark_layer_chunked(&dropout, ivy, state, true);
}
BENCHMARK(BM_Layer_Dropout_Ivy_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);

static void BM_Layer_Dropout_Ivy_Chunked_Backward(benchmark::State &state) {
    DropoutChunked dropout;
    benchmark_layer_chunked(&dropout, ivy, state, false);
}
BENCHMARK(BM_Layer_Dropout_Ivy_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);

// PIPELINED --- PIPELINED --- PIPELINED

static void BM_Layer_Dropout_Flickr_Pipelined_Forward(benchmark::State &state) {
    DropoutPipelined dropout;
    benchmark_layer_chunked(&dropout, flickr, state, true);
}
BENCHMARK(BM_Layer_Dropout_Flickr_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 15);

static void BM_Layer_Dropout_Flickr_Pipelined_Backward(benchmark::State &state) {
    DropoutPipelined dropout;
    benchmark_layer_chunked(&dropout, flickr, state, false);
}
BENCHMARK(BM_Layer_Dropout_Flickr_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 15);

static void BM_Layer_Dropout_Reddit_Pipelined_Forward(benchmark::State &state) {
    DropoutPipelined dropout;
    benchmark_layer_chunked(&dropout, reddit, state, true);
}
BENCHMARK(BM_Layer_Dropout_Reddit_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_Layer_Dropout_Reddit_Pipelined_Backward(benchmark::State &state) {
    DropoutPipelined dropout;
    benchmark_layer_chunked(&dropout, reddit, state, false);
}
BENCHMARK(BM_Layer_Dropout_Reddit_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_Layer_Dropout_Products_Pipelined_Forward(benchmark::State &state) {
    DropoutPipelined dropout;
    benchmark_layer_chunked(&dropout, products, state, true);
}
BENCHMARK(BM_Layer_Dropout_Products_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 21);

static void BM_Layer_Dropout_Products_Pipelined_Backward(benchmark::State &state) {
    DropoutPipelined dropout;
    benchmark_layer_chunked(&dropout, products, state, false);
}
BENCHMARK(BM_Layer_Dropout_Products_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 21);

static void BM_Layer_Dropout_Ivy_Pipelined_Forward(benchmark::State &state) {
    DropoutPipelined dropout;
    benchmark_layer_chunked(&dropout, ivy, state, true);
}
BENCHMARK(BM_Layer_Dropout_Ivy_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);

static void BM_Layer_Dropout_Ivy_Pipelined_Backward(benchmark::State &state) {
    DropoutPipelined dropout;
    benchmark_layer_chunked(&dropout, ivy, state, false);
}
BENCHMARK(BM_Layer_Dropout_Ivy_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);
