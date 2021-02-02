// Copyright 2021 Marcel Wagenländer

#include "memory_model.hpp"
#include "dataset.hpp"

#include <catch2/catch.hpp>
#include <iostream>
#include <math.h>

long test_composition_memory_model(CompositionMemoryModel *model, Dataset dataset) {
    long mib = 1 << 20;

    DatasetStats dataset_stats = get_dataset_stats(dataset);
    std::cout << "Dataset: " << get_dataset_name(dataset) << std::endl;
    std::cout << "Number of nodes: " << std::to_string(dataset_stats.num_nodes) << std::endl;
    std::cout << "Number of features: " << std::to_string(dataset_stats.num_features) << std::endl;

    long memory_usage = model->get_memory_usage();
    long memory_usage_mib = memory_usage * 4 / mib;
    std::cout << "Memory usage: " << std::to_string(memory_usage_mib) << " MiB" << std::endl;

    // GPU size MiB - cuda_helper MiB * (MiB) / bytes_per_elements;
    long max_num_elements = (16160 - 853) * mib / 4;
    long chunk_size = model->get_max_chunk_size(max_num_elements);
    std::cout << "Max chunk size: " << std::to_string(chunk_size) << " elements" << std::endl;
    long min_num_chunks = ceil((double) dataset_stats.num_nodes / (double) chunk_size);
    std::cout << "Min number of chunks: " << std::to_string(min_num_chunks) << std::endl;
    long chunk_size_min_chunks = ceil((double) dataset_stats.num_nodes / (double) min_num_chunks);
    std::cout << "Chunk size with min number of chunks: " << std::to_string(chunk_size_min_chunks) << " elements" << std::endl;

    MemoryUsageLayer memory_usage_layer = model->get_memory_usage_layer();
    long memory_usage_layer_mib = memory_usage_layer.memory_usage * 4 / mib;
    std::cout << "Layer memory usage: " << memory_usage_layer.layer->get_name() << ", ";
    std::cout << std::to_string(memory_usage_layer_mib) << " MiB" << std::endl;

    ChunkSizeLayer chunk_size_layer = model->get_max_chunk_size_layer(max_num_elements);
    std::cout << "Layer max chunk size: " << chunk_size_layer.layer->get_name() << ", ";
    std::cout << std::to_string(chunk_size_layer.chunk_size) << " elements" << std::endl;

    return 1;
}

long test_non_composition_memory_model(MemoryModel *model, Dataset dataset) {
    long mib = 1 << 20;

    DatasetStats dataset_stats = get_dataset_stats(dataset);
    std::cout << "Dataset: " << get_dataset_name(dataset) << std::endl;
    std::cout << "Number of nodes: " << std::to_string(dataset_stats.num_nodes) << std::endl;
    std::cout << "Number of features: " << std::to_string(dataset_stats.num_features) << std::endl;

    long memory_usage = model->get_memory_usage();
    long memory_usage_mib = memory_usage * 4 / mib;
    std::cout << "Memory usage: " << std::to_string(memory_usage_mib) << " MiB" << std::endl;

    // GPU size MiB - cuda_helper MiB * (MiB) / bytes_per_elements;
    long max_num_elements = (16160 - 853) * mib / 4;
    long chunk_size = model->get_max_chunk_size(max_num_elements);
    std::cout << "Max chunk size: " << std::to_string(chunk_size) << " elements" << std::endl;

    return 1;
}

TEST_CASE("Memory model, GraphSAGE", "[model][graphsage]") {
    // Model parameters
    long num_layers = 3;
    long num_hidden_channels = 256;

    std::vector<Dataset> datasets = {flickr, reddit, products, ivy};

    for (Dataset &dataset : datasets) {
        DatasetStats dataset_stats = get_dataset_stats(dataset);
        GraphSAGEMemoryModel model(num_layers, dataset_stats.num_nodes, dataset_stats.num_edges, dataset_stats.num_features,
                                   num_hidden_channels, dataset_stats.num_classes);

        CHECK(test_composition_memory_model(&model, dataset));
    }
}

TEST_CASE("Memory model, Dropout", "[model][dropout]") {
    std::vector<Dataset> datasets = {flickr, reddit, products, ivy};

    for (Dataset &dataset : datasets) {
        DatasetStats dataset_stats = get_dataset_stats(dataset);
        DropoutMemoryModel model(dataset_stats.num_nodes, dataset_stats.num_features);

        CHECK(test_non_composition_memory_model(&model, dataset));
    }
}
