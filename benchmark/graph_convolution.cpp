// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"
#include "memory.hpp"

#include <benchmark/benchmark.h>
#include <future>
#include <thread>

const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string reddit_dir_path = dir_path + "/reddit";
const std::string products_dir_path = dir_path + "/products";


static void BM_Graph_Convolution_Flickr_Forward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = flickr_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    CudaHelper cuda_helper;
    GraphConvolution graph_convolution(&cuda_helper, &adjacency, "mean", features.rows, features.columns);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/flickr_forward.log", std::move(future));

    matrix<float> *activations;
    for (auto _ : state) {
        activations = graph_convolution.forward(&features);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Convolution_Flickr_Forward);

static void BM_Graph_Convolution_Chunked_Flickr_Forward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = flickr_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    CudaHelper cuda_helper;
    GraphConvChunked graph_convolution(&cuda_helper, &adjacency, "mean", features.columns, state.range(0), features.rows);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/flickr_forward_" + std::to_string(state.range(0)) + ".log", std::move(future));

    matrix<float> *activations;
    for (auto _ : state) {
        activations = graph_convolution.forward(&features);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Convolution_Chunked_Flickr_Forward)->Range(1 << 10, 1 << 15);

static void BM_Graph_Convolution_Flickr_Backward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = flickr_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);
    matrix<float> in_gradients = gen_rand_matrix(features.rows, features.columns);

    CudaHelper cuda_helper;
    GraphConvolution graph_convolution(&cuda_helper, &adjacency, "mean", features.rows, features.columns);

    matrix<float> *activations = graph_convolution.forward(&features);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/flickr_backward.log", std::move(future));

   matrix<float> *gradients;
    for (auto _ : state) {
        gradients = graph_convolution.backward(&in_gradients);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Convolution_Flickr_Backward);

static void BM_Graph_Convolution_Flickr_Chunked_Backward(benchmark::State &state) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = flickr_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);
    matrix<float> in_gradients = gen_rand_matrix(features.rows, features.columns);

    CudaHelper cuda_helper;
    GraphConvChunked graph_convolution(&cuda_helper, &adjacency, "mean", features.columns, state.range(0), features.rows);

    matrix<float> *activations = graph_convolution.forward(&features);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/flickr_backward_" + std::to_string(state.range(0)) + ".log", std::move(future));

    matrix<float> *gradients;
    for (auto _ : state) {
        gradients = graph_convolution.backward(&in_gradients);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Convolution_Flickr_Chunked_Backward)->Range(1 << 10, 1 << 15);

static void BM_Graph_Convolution_Reddit_Forward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = reddit_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    CudaHelper cuda_helper;
    GraphConvolution graph_convolution(&cuda_helper, &adjacency, "mean", features.rows, features.columns);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/reddit_forward.log", std::move(future));

    matrix<float> *activations;
    for (auto _ : state) {
        activations = graph_convolution.forward(&features);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Convolution_Reddit_Forward);

static void BM_Graph_Convolution_Reddit_Chunked_Forward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = reddit_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    CudaHelper cuda_helper;
    GraphConvChunked graph_convolution(&cuda_helper, &adjacency, "mean", features.columns, state.range(0), features.rows);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/reddit_backward_" + std::to_string(state.range(0)) + ".log", std::move(future));

    matrix<float> *activations;
    for (auto _ : state) {
        activations = graph_convolution.forward(&features);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Convolution_Reddit_Chunked_Forward)->Range(1 << 10, 1 << 17);

static void BM_Graph_Reddit_Convolution_Backward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = reddit_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);
    matrix<float> in_gradients = gen_rand_matrix(features.rows, features.columns);

    CudaHelper cuda_helper;
    GraphConvolution graph_convolution(&cuda_helper, &adjacency, "mean", features.rows, features.columns);

    matrix<float> *activations = graph_convolution.forward(&features);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/reddit_backward.log", std::move(future));

    matrix<float> *gradients;
    for (auto _ : state) {
        gradients = graph_convolution.backward(&in_gradients);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Reddit_Convolution_Backward);

static void BM_Graph_Convolution_Reddit_Chunked_Backward(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = reddit_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);
    matrix<float> in_gradients = gen_rand_matrix(features.rows, features.columns);

    CudaHelper cuda_helper;
    GraphConvChunked graph_convolution(&cuda_helper, &adjacency, "mean", features.columns, state.range(0), features.rows);

    matrix<float> *activations = graph_convolution.forward(&features);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/reddit_backward_" + std::to_string(state.range(0)) + ".log", std::move(future));

    matrix<float> *gradients;
    for (auto _ : state) {
        gradients = graph_convolution.backward(&in_gradients);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Convolution_Reddit_Chunked_Backward)->Range(1 << 10, 1 << 17);

static void BM_Graph_Convolution_Products_Forward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = products_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    CudaHelper cuda_helper;
    GraphConvolution graph_convolution(&cuda_helper, &adjacency, "mean", features.rows, features.columns);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/products_forward.log", std::move(future));

    matrix<float> *activations;
    for (auto _ : state) {
        activations = graph_convolution.forward(&features);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Convolution_Products_Forward);

static void BM_Graph_Convolution_Products_Chunked_Forward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = products_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    CudaHelper cuda_helper;
    GraphConvChunked graph_convolution(&cuda_helper, &adjacency, "mean", features.columns, state.range(0), features.rows);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/products_backward_" + std::to_string(state.range(0)) + ".log", std::move(future));

    matrix<float> *activations;
    for (auto _ : state) {
        activations = graph_convolution.forward(&features);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Convolution_Products_Chunked_Forward)->Range(1 << 10, 1 << 21);

static void BM_Graph_Products_Convolution_Backward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = products_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);
    matrix<float> in_gradients = gen_rand_matrix(features.rows, features.columns);

    CudaHelper cuda_helper;
    GraphConvolution graph_convolution(&cuda_helper, &adjacency, "mean", features.rows, features.columns);

    matrix<float> *activations = graph_convolution.forward(&features);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/products_backward.log", std::move(future));

    matrix<float> *gradients;
    for (auto _ : state) {
        gradients = graph_convolution.backward(&in_gradients);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Products_Convolution_Backward);

static void BM_Graph_Convolution_Products_Chunked_Backward(benchmark::State &state) {
    std::string path;
    path = products_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    path = products_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);
    matrix<float> in_gradients = gen_rand_matrix(features.rows, features.columns);

    CudaHelper cuda_helper;
    GraphConvChunked graph_convolution(&cuda_helper, &adjacency, "mean", features.columns, state.range(0), features.rows);

    matrix<float> *activations = graph_convolution.forward(&features);

    std::promise<void> signal_exit;
    std::future<void> future = signal_exit.get_future();
    std::thread log_memory_thread(log_memory, "/tmp/products_backward_" + std::to_string(state.range(0)) + ".log", std::move(future));

    matrix<float> *gradients;
    for (auto _ : state) {
        gradients = graph_convolution.backward(&in_gradients);
    }

    signal_exit.set_value();
    log_memory_thread.join();
}
BENCHMARK(BM_Graph_Convolution_Products_Chunked_Backward)->Range(1 << 10, 1 << 21);
