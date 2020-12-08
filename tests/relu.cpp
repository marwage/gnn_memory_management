// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "helper.hpp"
#include "tensors.hpp"
#include "chunking.hpp"

#include "catch2/catch.hpp"
#include <string>

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
