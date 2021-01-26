// Copyright 2020 Marcel Wagenländer

#include "relu.hpp"
#include "chunking.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>

int test_layer(Layer *layer);
int test_layer_chunked(LayerChunked *layer, std::string py_name, long chunk_size, bool keep_allocation);


TEST_CASE("ReLU", "[relu]") {
    CudaHelper cuda_helper;
    // Flickr
    long num_nodes = 89250;
    long num_features = 500;
    Relu relu(&cuda_helper, num_nodes, num_features);
    CHECK(test_layer(&relu));
}

TEST_CASE("ReLU, chunked", "[relu][chunked]") {
    ReluChunked relu;
    CHECK(test_layer_chunked(&relu, "relu", 1 << 15, false));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 12, false));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 8, false));
}

TEST_CASE("ReLU, chunked, keep", "[relu][chunked][keep]") {
    ReluChunked relu;
    CHECK(test_layer_chunked(&relu, "relu", 1 << 15, true));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 12, true));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 8, true));
}

TEST_CASE("ReLU, pipelined", "[relu][pipelined]") {
    ReluPipelined relu;
    CHECK(test_layer_chunked(&relu, "relu", 1 << 15, false));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 12, false));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 8, false));
}
