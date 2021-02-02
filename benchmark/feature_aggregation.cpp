// Copyright 2020 Marcel Wagenländer

#include "feature_aggregation.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "dataset.hpp"
#include "gpu_memory_logger.hpp"
#include "sparse_computation.hpp"
#include "tensors.hpp"

#include <benchmark/benchmark.h>

const std::string dir_path = "/mnt/data";


void benchmark_feature_aggregation(Dataset dataset, benchmark::State &state, bool forward) {
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);
    std::string path = dataset_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);
    path = dataset_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);
    Matrix<float> adjacency_row_sum(features.num_rows_, 1, true);
    sp_mat_sum_rows(&adjacency, &adjacency_row_sum);
    Matrix<float> incoming_gradients;
    if (!forward) {
        incoming_gradients.set(features.num_rows_, features.num_columns_, true);
        incoming_gradients.set_random_values();
    }

    CudaHelper cuda_helper;
    FeatureAggregation feature_aggr(&cuda_helper, features.num_rows_, features.num_columns_, &adjacency, mean, &adjacency_row_sum);

    if (!forward) {
        feature_aggr.forward(&features);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(feature_aggr.get_name() + "_" + get_dataset_name(dataset) + "_" + direction);
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

void benchmark_feature_aggregation_chunked(FeatureAggregationChunked *feature_aggr, Dataset dataset, benchmark::State &state, bool forward) {
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);
    std::string path = dataset_path + "/features.npy";
    Matrix<float> *features = new Matrix<float>();
    load_npy_matrix<float>(path, features);
    to_column_major_inplace(features);
    long num_nodes = features->num_rows_;
    long num_features = features->num_columns_;
    path = dataset_path + "/adjacency.mtx";
    SparseMatrix<float> *adjacency = new SparseMatrix<float>();
    load_mtx_matrix<float>(path, adjacency);

    long chunk_size = state.range(0);
    long num_chunks = ceil((double) num_nodes / (double) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(features, &features_chunked, chunk_size);
    delete features;
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    if (!forward) {
        init_set_random_values(&incoming_gradients_chunked, num_nodes, num_features, chunk_size, false);
    }

    // chunk adjacency
    std::vector<SparseMatrix<float>> adjacencies(num_chunks * num_chunks);
    double_chunk_up_sp(adjacency, &adjacencies, chunk_size);
    Matrix<float> adjacency_row_sum(num_nodes, 1, true);
    sp_mat_sum_rows(adjacency, &adjacency_row_sum);
    delete adjacency;

    CudaHelper cuda_helper;
    feature_aggr->set(&cuda_helper, &adjacencies, &adjacency_row_sum, "mean", num_features, chunk_size, num_nodes);

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

static void BM_Layer_FeatureAggregation_Layer_Flickr_Forward(benchmark::State &state) {
    benchmark_feature_aggregation(flickr, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Layer_Flickr_Forward);

static void BM_Layer_FeatureAggregation_Layer_Flickr_Backward(benchmark::State &state) {
    benchmark_feature_aggregation(flickr, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Layer_Flickr_Backward);

static void BM_Layer_FeatureAggregation_Layer_Reddit_Forward(benchmark::State &state) {
    benchmark_feature_aggregation(reddit, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Layer_Reddit_Forward);

static void BM_Layer_FeatureAggregation_Layer_Reddit_Backward(benchmark::State &state) {
    benchmark_feature_aggregation(reddit, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Layer_Reddit_Backward);

static void BM_Layer_FeatureAggregation_Layer_Products_Forward(benchmark::State &state) {
    benchmark_feature_aggregation(products, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Layer_Products_Forward);

static void BM_Layer_FeatureAggregation_Layer_Products_Backward(benchmark::State &state) {
    benchmark_feature_aggregation(products, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Layer_Products_Backward);

// CHUNKED --- CHUNKED --- CHUNKED

static void BM_Layer_FeatureAggregation_Flickr_Chunked_Forward(benchmark::State &state) {
    FeatureAggregationChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, flickr, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Flickr_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 14, 1 << 16);

static void BM_Layer_FeatureAggregation_Flickr_Chunked_Backward(benchmark::State &state) {
    FeatureAggregationChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, flickr, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Flickr_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 14, 1 << 16);

static void BM_Layer_FeatureAggregation_Reddit_Chunked_Forward(benchmark::State &state) {
    FeatureAggregationChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, reddit, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Reddit_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 14, 1 << 17);

static void BM_Layer_FeatureAggregation_Reddit_Chunked_Backward(benchmark::State &state) {
    FeatureAggregationChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, reddit, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Reddit_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 14, 1 << 17);

static void BM_Layer_FeatureAggregation_Products_Chunked_Forward(benchmark::State &state) {
    FeatureAggregationChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, products, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Products_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 14, 1 << 21);

static void BM_Layer_FeatureAggregation_Products_Chunked_Backward(benchmark::State &state) {
    FeatureAggregationChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, products, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Products_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 14, 1 << 21);

static void BM_Layer_FeatureAggregation_Ivy_Chunked_Forward(benchmark::State &state) {
    FeatureAggregationChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, ivy, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Ivy_Chunked_Forward)->RangeMultiplier(2)->Range(1 << 14, 1 << 19);

static void BM_Layer_FeatureAggregation_Ivy_Chunked_Backward(benchmark::State &state) {
    FeatureAggregationChunked feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, ivy, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Ivy_Chunked_Backward)->RangeMultiplier(2)->Range(1 << 14, 1 << 19);

// PIPELINED --- PIPELINED --- PIPELINED

static void BM_Layer_FeatureAggregation_Flickr_Pipelined_Forward(benchmark::State &state) {
    FeatureAggregationPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, flickr, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Flickr_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 14, 1 << 16);

static void BM_Layer_FeatureAggregation_Flickr_Pipelined_Backward(benchmark::State &state) {
    FeatureAggregationPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, flickr, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Flickr_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 14, 1 << 16);

static void BM_Layer_FeatureAggregation_Reddit_Pipelined_Forward(benchmark::State &state) {
    FeatureAggregationPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, reddit, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Reddit_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 14, 1 << 17);

static void BM_Layer_FeatureAggregation_Reddit_Pipelined_Backward(benchmark::State &state) {
    FeatureAggregationPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, reddit, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Reddit_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 14, 1 << 17);

static void BM_Layer_FeatureAggregation_Products_Pipelined_Forward(benchmark::State &state) {
    FeatureAggregationPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, products, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Products_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 14, 1 << 21);

static void BM_Layer_FeatureAggregation_Products_Pipelined_Backward(benchmark::State &state) {
    FeatureAggregationPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, products, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Products_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 14, 1 << 21);

static void BM_Layer_FeatureAggregation_Ivy_Pipelined_Forward(benchmark::State &state) {
    FeatureAggregationPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, ivy, state, true);
}
BENCHMARK(BM_Layer_FeatureAggregation_Ivy_Pipelined_Forward)->RangeMultiplier(2)->Range(1 << 14, 1 << 19);

static void BM_Layer_FeatureAggregation_Ivy_Pipelined_Backward(benchmark::State &state) {
    FeatureAggregationPipelined feature_aggr;
    benchmark_feature_aggregation_chunked(&feature_aggr, ivy, state, false);
}
BENCHMARK(BM_Layer_FeatureAggregation_Ivy_Pipelined_Backward)->RangeMultiplier(2)->Range(1 << 14, 1 << 19);
