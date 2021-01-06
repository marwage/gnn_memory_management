// Copyright 2020 Marcel Wagenländer

#include "relu.hpp"
#include "chunking.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>

const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";

int test_layer(Layer *layer, std::string py_name);
int test_layer_chunked(LayerChunked *layer, std::string py_name, long chunk_size);


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

TEST_CASE("ReLU, pipelined", "[relu][pipelined]") {
    ReluPipelined relu;
    CHECK(test_layer_chunked(&relu, "relu", 1 << 15));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 12));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 8));
}
