// Copyright 2020 Marcel Wagenländer

#include "relu.hpp"
#include "chunking.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>

int test_layer(Layer *layer, std::string py_name);
int test_layer_chunked(LayerChunked *layer, std::string py_name, long chunk_size);
int test_layer_chunked_keep(LayerChunked *layer, std::string py_name, long chunk_size);


TEST_CASE("ReLU", "[relu]") {
    Relu relu;
    CHECK(test_layer(&relu, "relu"));
}

TEST_CASE("ReLU, chunked", "[relu][chunked]") {
    ReluChunked relu;
    CHECK(test_layer_chunked(&relu, "relu", 1 << 15));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 12));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 8));
}

TEST_CASE("ReLU, chunked, keep", "[relu][chunked][keep]") {
    ReluChunked relu;
    CHECK(test_layer_chunked_keep(&relu, "relu", 1 << 15));
    CHECK(test_layer_chunked_keep(&relu, "relu", 1 << 12));
    CHECK(test_layer_chunked_keep(&relu, "relu", 1 << 8));
}

TEST_CASE("ReLU, pipelined", "[relu][pipelined]") {
    ReluPipelined relu;
    CHECK(test_layer_chunked(&relu, "relu", 1 << 15));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 12));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 8));
}
