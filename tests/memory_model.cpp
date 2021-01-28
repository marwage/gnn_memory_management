// Copyright 2021 Marcel Wagenländer

#include "memory_model.hpp"

#include <catch2/catch.hpp>
#include <iostream>

long test_graphsage_memory_model() {
    long num_layers = 3;
    long num_hidden_channels = 256;

    // Flickr
//    long num_nodes = 89250;
//    long num_edges = 899756;
//    long num_features = 500;
//    long num_classes = 7;

    // Products
    long num_nodes = 2449029;
    long num_edges = 61859140;
    long num_features = 100;
    long num_classes = 47;

    GraphSAGEMemoryModel model(num_layers, num_nodes, num_edges, num_features, num_hidden_channels, num_classes);

    long max_memory = model.get_memory_usage();
    std::cout << "Max memory: " << std::to_string(max_memory) << std::endl;

    // GPU size MiB - cuda_helper MiB * (MiB) / bytes_per_elements;
    long mib = 1 << 20;
    long max_num_elements = (16160 - 853) * mib / 4;

    long chunk_size = model.get_max_chunk_size(max_num_elements);

    std::cout << "Max chunk size: " << std::to_string(chunk_size) << std::endl;

    return chunk_size;
}

TEST_CASE("Memory model, GraphSAGE", "[model][graphsage]") {
    test_graphsage_memory_model();
}
