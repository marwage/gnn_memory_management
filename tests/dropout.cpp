// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"

#include "catch2/catch.hpp"
#include <string>

int test_layer(Layer *layer, std::string py_name);
int test_layer_chunked(LayerChunked *layer, std::string py_name, long chunk_size);


TEST_CASE("Dropout", "[dropout]") {
    Dropout dropout;
    CHECK(test_layer(&dropout, "dropout"));
}

TEST_CASE("Dropout chunked", "[dropout][chunked]") {
    DropoutChunked dropout;
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 15));
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 12));
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 8));
}
