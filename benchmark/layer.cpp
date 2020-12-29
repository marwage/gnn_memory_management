// Copyright 2020 Marcel Wagenländer

#include "layer.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "dataset.hpp"
#include "gpu_memory_logger.hpp"
#include "tensors.hpp"

#include <benchmark/benchmark.h>

const std::string dir_path = "/mnt/data";


void benchmark_layer(Layer *layer, Dataset dataset, benchmark::State &state, bool forward) {
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);
    std::string path;
    path = dataset_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> incoming_gradients;
    if (!forward) {
        incoming_gradients.set(features.num_rows_, features.num_columns_, true);
        incoming_gradients.set_random_values();
    }

    CudaHelper cuda_helper;
    layer->set(&cuda_helper, features.num_rows_, features.num_columns_);

    if (!forward) {
        layer->forward(&features);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(layer->name_ + "_" + get_dataset_name(dataset) + "_" + direction);
    memory_logger.start();

    for (auto _ : state) {
        if (forward) {
            layer->forward(&features);
        } else {
            layer->backward(&incoming_gradients);
        }
    }

    memory_logger.stop();
}

void benchmark_layer_chunked(LayerChunked *layer, Dataset dataset, benchmark::State &state, bool forward) {
    std::string dataset_path = dir_path + "/" + get_dataset_name(dataset);
    std::string path;
    path = dataset_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
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
    chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);

    CudaHelper cuda_helper;
    layer->set(&cuda_helper, state.range(0), features.num_rows_, features.num_columns_);

    if (!forward) {
        layer->forward(&features_chunked);
    }

    std::string direction;
    if (forward) {
        direction = "forward";
    } else {
        direction = "backward";
    }

    GPUMemoryLogger memory_logger(layer->name_ + "_" + get_dataset_name(dataset) + "_" + direction + "_" + std::to_string(chunk_size));
    memory_logger.start();

    for (auto _ : state) {
        if (forward) {
            layer->forward(&features_chunked);
        } else {
            layer->backward(&incoming_gradients_chunked);
        }
    }

    memory_logger.stop();
}
