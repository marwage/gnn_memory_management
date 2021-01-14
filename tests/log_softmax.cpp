// Copyright 2020 Marcel Wagenländer

#include "log_softmax.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <stdlib.h>

int test_layer(Layer *layer, std::string py_name);
int test_layer_chunked(LayerChunked *layer, std::string py_name, long chunk_size);


TEST_CASE("Log-softmax", "[logsoftmax]") {
    LogSoftmax logsoftmax;
    CHECK(test_layer(&logsoftmax, "log_softmax"));
}

TEST_CASE("Log-softmax, chunked", "[logsoftmax][chunked]") {
    LogSoftmaxChunked logsoftmax;
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 15));
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 12));
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 8));
}

TEST_CASE("Log-softmax, piplined", "[logsoftmax][piplined]") {
    LogSoftmaxPipelined logsoftmax;
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 15));
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 12));
    CHECK(test_layer_chunked(&logsoftmax, "log_softmax", 1 << 8));
}