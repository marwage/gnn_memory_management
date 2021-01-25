// Copyright 2020 Marcel Wagenländer

#include "log_softmax.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <stdlib.h>

int test_layer(Layer *layer, std::string py_name);
int test_layer_chunked(LayerChunked *layer, std::string py_name, long chunk_size, bool keep_allocation);


TEST_CASE("Log-softmax", "[logsoftmax]") {
    LogSoftmax logsoftmax;
    CHECK(test_layer(&logsoftmax, "log_softmax"));
}

TEST_CASE("Log-softmax, chunked", "[logsoftmax][chunked]") {
    LogSoftmaxChunked logsoftmax;
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 15, false));
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 12, false));
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 8, false));
}

TEST_CASE("Log-softmax, chunked, keep", "[logsoftmax][chunked][keep]") {
    LogSoftmaxChunked logsoftmax;
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 15, true));
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 12, true));
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 8, true));
}

TEST_CASE("Log-softmax, piplined", "[logsoftmax][piplined]") {
    LogSoftmaxPipelined logsoftmax;
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 15, false));
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 12, false));
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 8, false));
}
