// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
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

const long num_out_features = 256;


void benchmark_sagelinear(Dataset dataset, benchmark::State &state, bool forward) {
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
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, false);
    aggregates.set_random_values();
    Matrix<float> incoming_gradients;
    if (!forward) {
        incoming_gradients.set(features.num_rows_, features.num_columns_, true);
        incoming_gradients.set_random_values();
    }

    CudaHelper cuda_helper;
    SageLinear sagelinear(&cuda_helper, features.num_columns_, num_out_features, features.num_rows_);

    if (!forward) {
        sagelinear.forward(&features, &aggregates);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(sagelinear.name_ + "_" + get_dataset_name(dataset) + "_" + direction);
    memory_logger.start();

    for (auto _ : state) {
        if (forward) {
            sagelinear.forward(&features, &aggregates);
        } else {
            sagelinear.backward(&incoming_gradients);
        }
    }

    memory_logger.stop();
}

void benchmark_sagelinear_chunked(SageLinearChunkedParent *sagelinear, Dataset dataset, benchmark::State &state, bool forward) {
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
    Matrix<float> aggregates(features.num_rows_, features.num_columns_, false);
    aggregates.set_random_values();
    Matrix<float> incoming_gradients;
    if (!forward) {
        incoming_gradients.set(features.num_rows_, features.num_columns_, true);
        incoming_gradients.set_random_values();
    }

    long chunk_size = state.range(0);
    long num_chunks = ceil((float) features.num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    std::vector<Matrix<float>> aggregates_chunked(num_chunks);
    chunk_up(&aggregates, &aggregates_chunked, chunk_size);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    if (!forward) {
        chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);
    }

    CudaHelper cuda_helper;
    sagelinear->set(&cuda_helper, features.num_columns_, num_out_features, chunk_size, features.num_rows_);

    if (!forward) {
        sagelinear->forward(&features_chunked, &aggregates_chunked);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(sagelinear->name_ + "_" + get_dataset_name(dataset) + "_" + direction + "_" + std::to_string(chunk_size));
    memory_logger.start();

    for (auto _ : state) {
        if (forward) {
            sagelinear->forward(&features_chunked, &aggregates_chunked);
        } else {
            sagelinear->backward(&incoming_gradients_chunked);
        }
    }

    memory_logger.stop();
}


static void BM_Layer_Sagelinear_Flickr_Forward(benchmark::State &state) {
    benchmark_sagelinear(flickr, state, true);
}
BENCHMARK(BM_Layer_Sagelinear_Flickr_Forward);

static void BM_Layer_Sagelinear_Flickr_Backward(benchmark::State &state) {
    benchmark_sagelinear(flickr, state, false);
}
BENCHMARK(BM_Layer_Sagelinear_Flickr_Backward);


static void BM_Layer_Sagelinear_Reddit_Forward(benchmark::State &state) {
    benchmark_sagelinear(reddit, state, true);
}
BENCHMARK(BM_Layer_Sagelinear_Reddit_Forward);

static void BM_Layer_Sagelinar_Reddit_Backward(benchmark::State &state) {
    benchmark_sagelinear(reddit, state, false);
}
BENCHMARK(BM_Layer_Sagelinar_Reddit_Backward);

static void BM_Layer_Sagelinear_Products_Forward(benchmark::State &state) {
    benchmark_sagelinear(products, state, true);
}
BENCHMARK(BM_Layer_Sagelinear_Products_Forward);

static void BM_Layer_Sagelinear_Products_Backward(benchmark::State &state) {
    benchmark_sagelinear(products, state, false);
}
BENCHMARK(BM_Layer_Sagelinear_Products_Backward);

// CHUNKED --- CHUNKED --- CHUNKED

static void BM_Layer_Sagelinear_Flickr_Chunked_Forward(benchmark::State &state) {
    SageLinearChunked sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, flickr, state, true);
}
BENCHMARK(BM_Layer_Sagelinear_Flickr_Chunked_Forward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Sagelinear_Flickr_Chunked_Backward(benchmark::State &state) {
    SageLinearChunked sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, flickr, state, false);
}
BENCHMARK(BM_Layer_Sagelinear_Flickr_Chunked_Backward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Sagelinear_Reddit_Chunked_Forward(benchmark::State &state) {
    SageLinearChunked sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, reddit, state, true);
}
BENCHMARK(BM_Layer_Sagelinear_Reddit_Chunked_Forward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Sagelinear_Reddit_Chunked_Backward(benchmark::State &state) {
    SageLinearChunked sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, reddit, state, false);
}
BENCHMARK(BM_Layer_Sagelinear_Reddit_Chunked_Backward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Sagelinear_Products_Chunked_Forward(benchmark::State &state) {
    SageLinearChunked sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, products, state, true);
}
BENCHMARK(BM_Layer_Sagelinear_Products_Chunked_Forward)->Range(1 << 16, 1 << 21);

static void BM_Layer_Sagelinear_Products_Chunked_Backward(benchmark::State &state) {
    SageLinearChunked sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, products, state, false);
}
BENCHMARK(BM_Layer_Sagelinear_Products_Chunked_Backward)->Range(1 << 16, 1 << 21);

// PIPELINED --- PIPELINED --- PIPELINED

static void BM_Layer_Sagelinear_Flickr_Pipelined_Forward(benchmark::State &state) {
    SageLinearPipelined sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, flickr, state, true);
}
BENCHMARK(BM_Layer_Sagelinear_Flickr_Pipelined_Forward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Sagelinear_Flickr_Pipelined_Backward(benchmark::State &state) {
    SageLinearPipelined sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, flickr, state, false);
}
BENCHMARK(BM_Layer_Sagelinear_Flickr_Pipelined_Backward)->Range(1 << 10, 1 << 15);

static void BM_Layer_Sagelinear_Reddit_Pipelined_Forward(benchmark::State &state) {
    SageLinearPipelined sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, reddit, state, true);
}
BENCHMARK(BM_Layer_Sagelinear_Reddit_Pipelined_Forward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Sagelinear_Reddit_Pipelined_Backward(benchmark::State &state) {
    SageLinearPipelined sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, reddit, state, false);
}
BENCHMARK(BM_Layer_Sagelinear_Reddit_Pipelined_Backward)->Range(1 << 12, 1 << 17);

static void BM_Layer_Sagelinear_Products_Pipelined_Forward(benchmark::State &state) {
    SageLinearPipelined sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, products, state, true);
}
BENCHMARK(BM_Layer_Sagelinear_Products_Pipelined_Forward)->Range(1 << 16, 1 << 21);

static void BM_Layer_Sagelinear_Products_Pipelined_Backward(benchmark::State &state) {
    SageLinearPipelined sagelinear;
    benchmark_sagelinear_chunked(&sagelinear, products, state, false);
}
BENCHMARK(BM_Layer_Sagelinear_Products_Pipelined_Backward)->Range(1 << 16, 1 << 21);
