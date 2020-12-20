// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "dataset.hpp"
#include "gpu_memory_logger.hpp"
#include "tensors.hpp"

#include <benchmark/benchmark.h>

const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string reddit_dir_path = dir_path + "/reddit";
const std::string products_dir_path = dir_path + "/products";


void benchmark_feature_aggregation(Dataset dataset, benchmark::State &state, bool forward) {
    std::string dataset_path;
    if (dataset == flickr) {
        dataset_path = flickr_dir_path;
    } else if (dataset == reddit) {
        dataset_path = reddit_dir_path;
    } else if (dataset == products) {
        dataset_path = products_dir_path;
    }
    std::string path = dataset_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    path = flickr_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);
    Matrix<float> incoming_gradients;
    if (!forward) {
        incoming_gradients.set(features.num_rows_, features.num_columns_, true);
        incoming_gradients.set_random_values();
    }

    CudaHelper cuda_helper;
    GraphConvolution feature_aggr(&cuda_helper, &adjacency, "mean", features.num_rows_, features.num_columns_);

    if (!forward) {
        feature_aggr.forward(&features);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(feature_aggr.name_ + "_" + get_dataset_name(dataset) + "_" + direction);
    memory_logger.start();

    for (auto _ : state) {
        if (forward) {
            feature_aggr.forward(&features);
        } else {
            feature_aggr.backward(&incoming_gradients);
        }
    }

    memory_logger.stop();
}

void benchmark_feature_aggregation_chunked(GraphConvChunked *feature_aggr, Dataset dataset, benchmark::State &state, bool forward) {
    std::string dataset_path;
    if (dataset == flickr) {
        dataset_path = flickr_dir_path;
    } else if (dataset == reddit) {
        dataset_path = reddit_dir_path;
    } else if (dataset == products) {
        dataset_path = products_dir_path;
    }
    std::string path = dataset_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    path = flickr_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);
    Matrix<float> incoming_gradients;
    if (!forward) {
        incoming_gradients.set(features.num_rows_, features.num_columns_, true);
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
    feature_aggr->set(&cuda_helper, &adjacency, "mean", features.num_columns_, chunk_size, features.num_rows_);
    ;

    if (!forward) {
        feature_aggr->forward(&features_chunked);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(feature_aggr->name_ + "_" + get_dataset_name(dataset) + "_" + direction + "_" + std::to_string(chunk_size));
    memory_logger.start();

    for (auto _ : state) {
        if (forward) {
            feature_aggr->forward(&features_chunked);
        } else {
            feature_aggr->backward(&incoming_gradients_chunked);
        }
    }

    memory_logger.stop();
}

static void BM_Layer_Graph_Convolution_Chunked_Reddit_Constructor(benchmark::State &state) {
    std::string path;
    path = reddit_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);
    long num_nodes = adjacency.num_rows_;
    long num_features = 602;// reddit dataset
    CudaHelper cuda_helper;

    GPUMemoryLogger memory_logger("graphconv_reddit_constructor_" + std::to_string(state.range(0)));
    memory_logger.start();

    for (auto _ : state) {
        GraphConvChunked graph_convolution(&cuda_helper, &adjacency, "mean", num_features, state.range(0), num_nodes);
    }

    memory_logger.stop();
}
BENCHMARK(BM_Layer_Graph_Convolution_Chunked_Reddit_Constructor)->Range(1 << 12, 1 << 17);

static void BM_Layer_Graph_Convolution_Flickr_Forward(benchmark::State &state) {
    benchmark_feature_aggregation(flickr, state, true);
}
BENCHMARK(BM_Layer_Graph_Convolution_Flickr_Forward);

static void BM_Layer_Graph_Convolution_Flickr_Backward(benchmark::State &state) {
    benchmark_feature_aggregation(flickr, state, false);
}
BENCHMARK(BM_Layer_Graph_Convolution_Flickr_Backward);

static void BM_Layer_Graph_Convolution_Reddit_Forward(benchmark::State &state) {
    benchmark_feature_aggregation(reddit, state, true);
}
BENCHMARK(BM_Layer_Graph_Convolution_Reddit_Forward);

static void BM_Layer_Graph_Convolution_Reddit_Backward(benchmark::State &state) {
    benchmark_feature_aggregation(reddit, state, false);
}
BENCHMARK(BM_Layer_Graph_Convolution_Reddit_Backward);

static void BM_Layer_Graph_Convolution_Products_Forward(benchmark::State &state) {
    benchmark_feature_aggregation(products, state, true);
}
BENCHMARK(BM_Layer_Graph_Convolution_Products_Forward);

static void BM_Layer_Graph_Convolution_Products_Backward(benchmark::State &state) {
    benchmark_feature_aggregation(products, state, false);
}
BENCHMARK(BM_Layer_Graph_Convolution_Products_Backward);

// CHUNKED --- CHUNKED --- CHUNKED

static void BM_Layer_Graph_Convolution_Flickr_Chunked_Forward(benchmark::State &state) {
    GraphConvChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, flickr, state, true);
}
BENCHMARK(BM_Layer_Graph_Convolution_Flickr_Chunked_Forward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Graph_Convolution_Flickr_Chunked_Backward(benchmark::State &state) {
    GraphConvChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, flickr, state, false);
}
BENCHMARK(BM_Layer_Graph_Convolution_Flickr_Chunked_Backward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Graph_Convolution_Reddit_Chunked_Forward(benchmark::State &state) {
    GraphConvChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, reddit, state, true);
}
BENCHMARK(BM_Layer_Graph_Convolution_Reddit_Chunked_Forward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Graph_Convolution_Reddit_Chunked_Backward(benchmark::State &state) {
    GraphConvChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, reddit, state, false);
}
BENCHMARK(BM_Layer_Graph_Convolution_Reddit_Chunked_Backward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Graph_Convolution_Products_Chunked_Forward(benchmark::State &state) {
    GraphConvChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, products, state, true);
}
BENCHMARK(BM_Layer_Graph_Convolution_Products_Chunked_Forward)->Range(1 << 16, 1 << 21);

static void BM_Layer_Graph_Convolution_Products_Chunked_Backward(benchmark::State &state) {
    GraphConvChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, products, state, false);
}
BENCHMARK(BM_Layer_Graph_Convolution_Products_Chunked_Backward)->Range(1 << 16, 1 << 21);

// PIPELINED --- PIPELINED --- PIPELINED

static void BM_Layer_Graph_Convolution_Flickr_Pipelined_Forward(benchmark::State &state) {
    GraphConvPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, flickr, state, true);
}
BENCHMARK(BM_Layer_Graph_Convolution_Flickr_Pipelined_Forward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Graph_Convolution_Flickr_Pipelined_Backward(benchmark::State &state) {
    GraphConvPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, flickr, state, false);
}
BENCHMARK(BM_Layer_Graph_Convolution_Flickr_Pipelined_Backward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Graph_Convolution_Reddit_Pipelined_Forward(benchmark::State &state) {
    GraphConvPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, reddit, state, true);
}
BENCHMARK(BM_Layer_Graph_Convolution_Reddit_Pipelined_Forward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Graph_Convolution_Reddit_Pipelined_Backward(benchmark::State &state) {
    GraphConvPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, reddit, state, false);
}
BENCHMARK(BM_Layer_Graph_Convolution_Reddit_Pipelined_Backward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Graph_Convolution_Products_Pipelined_Forward(benchmark::State &state) {
    GraphConvPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, products, state, true);
}
BENCHMARK(BM_Layer_Graph_Convolution_Products_Pipelined_Forward)->Range(1 << 16, 1 << 21);

static void BM_Layer_Graph_Convolution_Products_Pipelined_Backward(benchmark::State &state) {
    GraphConvPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, products, state, false);
}
BENCHMARK(BM_Layer_Graph_Convolution_Products_Pipelined_Backward)->Range(1 << 16, 1 << 21);
