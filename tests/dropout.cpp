// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "chunking.hpp" // TEMPORARY COPY AND PASTE
#include "helper.hpp" // TEMPORARY COPY AND PASTE

#include "catch2/catch.hpp"
#include <cudnn.h>
#include <fstream>
#include <iostream>
#include <string>

int test_layer(Layer *layer, std::string py_name);
int test_layer_chunked(LayerChunked *layer, std::string py_name, long chunk_size, bool keep_allocation);


TEST_CASE("Dropout", "[dropout]") {
    Dropout dropout;
    CHECK(test_layer(&dropout, "dropout"));
}

TEST_CASE("Dropout, chunked", "[dropout][chunked]") {
    DropoutChunked dropout;
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 15, false));
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 12, false));
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 8, false));
}

TEST_CASE("Dropout, chunked, keep", "[dropout][chunked][keep]") {
    DropoutChunked dropout;
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 15, true));
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 12, true));
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 8, true));
}

TEST_CASE("Dropout, pipelined", "[dropout][pipelined]") {
    DropoutPipelined dropout;
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 15, false));
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 12, false));
    CHECK(test_layer_chunked(&dropout, "dropout", 1 << 8, false));
}

TEST_CASE("Dropout, memory", "[dropout][memory]") {
    CudaHelper helper;
    size_t state_size;
    size_t reserve_space_size;
    cudnnTensorDescriptor_t x_descr;
    std::ofstream csvfile;

    check_cudnn(cudnnDropoutGetStatesSize(helper.cudnn_handle, &state_size));

    std::cout << "State size: " << state_size << std::endl;

    csvfile.open ("/tmp/reserve_space_size.csv");
    csvfile << "num_rows,num_cols,reserve_space_size\n";
    long max_row_col = 1 << 25;
    long max_size = 1;
    max_size = max_size << 31;
    for (long num_rows = 1000; num_rows < max_row_col; num_rows = num_rows + 1000) {
        for (long num_cols = 1000; num_cols < max_row_col; num_cols = num_cols + 1000) {
            if (num_rows * num_cols < max_size) {
                check_cudnn(cudnnCreateTensorDescriptor(&x_descr));
                check_cudnn(cudnnSetTensor4dDescriptor(x_descr,
                                                       CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                       num_rows, 1, 1, num_cols));

                check_cudnn(cudnnDropoutGetReserveSpaceSize(x_descr, &reserve_space_size));

                csvfile << std::to_string(num_rows) + "," + std::to_string(num_cols) + "," + std::to_string(reserve_space_size) + "\n";

                check_cudnn(cudnnDestroyTensorDescriptor(x_descr));
            }
        }
    }

    csvfile.close();
}
